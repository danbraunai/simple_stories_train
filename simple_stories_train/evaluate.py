import re

from inspect_ai import Task, eval, task
from inspect_ai.dataset import FieldSpec, Sample, hf_dataset
from inspect_ai.model import GenerateConfig, get_model, modelapi
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate


def record_to_sample(record: dict[str, str]) -> Sample:
    return Sample(input=record["story"], id=record.get("id", ""))


dataset = hf_dataset(
    "lennart-finke/SimpleStories",
    split="test",
    sample_fields=FieldSpec(input="story", target="story"),
    limit=1000,
)


@scorer(metrics={"originality": [accuracy()], "coherence": [accuracy()], "grammar": [accuracy()]})
def story_quality_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        judge_model = get_model("openai/gpt-4o-mini")
        generated_story = state.output.completion + "..."

        # Extract the final answer after chain of thought
        answer_match = re.search(r"ANSWER:(.*?)(?=ANSWER:|$)", generated_story, re.DOTALL)
        story = answer_match.group(1).strip() if answer_match else generated_story

        # Craft prompt for the judge
        evaluation_prompt = f"""
Evaluate the following story based on three criteria by assigning each a score from 0 to 100:
1. **Originality**: Rate the creativity and uniqueness of the story.
2. **Coherence**: Rate the logical flow and consistency of the story.
3. **Grammar**: Rate the grammatical correctness of the story. Ignore spacing and capitalization.

**Story to evaluate:**
{story}

Please provide your assessment in the following format, ensuring each score is an integer between 0 and 100:
{{"ORIGINALITY": 0, "COHERENCE": 0, "GRAMMAR": 0}}
"""

        # Get evaluation from judge model
        result = await judge_model.generate(
            evaluation_prompt, config=GenerateConfig(temperature=0.0)
        )

        # Use regex to find a dictionary
        dict_match = re.search(
            r'\{.*"ORIGINALITY":\s*(\d+).*"COHERENCE":\s*(\d+).*"GRAMMAR":\s*(\d+).*\}',
            result.completion,
        )

        if dict_match:
            scores = {
                "originality": int(dict_match.group(1)),
                "coherence": int(dict_match.group(2)),
                "grammar": int(dict_match.group(3)),
            }
        else:
            scores = {"originality": 0, "coherence": 0, "grammar": 0}

        scores = {k: max(0, min(v, 100)) for k, v in scores.items()}

        return Score(
            value=scores,  # type:ignore
            answer=story,
            explanation=result.completion,
        )

    return score


@modelapi(name="simple_stories")
def simple_stories():
    from inspect_api import SimpleStoriesAPI

    return SimpleStoriesAPI


@task
def evaluate_story_generation():
    """Task definition for evaluating story generation capabilities."""
    return Task(
        dataset=dataset,
        plan=[generate()],
        scorer=story_quality_scorer(),
    )


if __name__ == "__main__":
    model = get_model(
        "simple_stories/lennart-finke/SimpleStories-125M",
        model_path="lennart-finke/SimpleStories-125M",
    )
    eval(evaluate_story_generation, model=model, limit=1, max_tokens=100)
