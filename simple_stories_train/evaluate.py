import re

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig, get_model, modelapi
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState, generate


def record_to_sample(record: dict[str, str]) -> Sample:
    return Sample(input=record["story"].split("\n")[0], id=record.get("generation_id", ""))


dataset = hf_dataset(
    "lennart-finke/SimpleStories",
    split="test",
    limit=1000,
    sample_fields=record_to_sample,
)


@scorer(
    metrics={
        "originality": [mean()],
        "coherence": [mean()],
        "grammar": [mean()],
        "quality": [mean()],
    }
)
def story_quality_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        judge_model = get_model("openai/gpt-4o-mini")
        generated_story = state.output.completion + "..."

        # Extract the final answer after chain of thought
        answer_match = re.search(r"ANSWER:(.*?)(?=ANSWER:|$)", generated_story, re.DOTALL)
        story = answer_match.group(1).strip() if answer_match else generated_story

        # Craft prompt for the judge
        evaluation_prompt = f"""
Evaluate the following story based on four criteria by assigning each a score from 0 to 100:
1. **Originality**: Rate the creativity and uniqueness of the story.
2. **Coherence**: Rate the logical flow and consistency of the story.
3. **Grammar**: Rate the grammatical correctness of the story. Ignore spacing and capitalization.
4. **Quality**: Rate the overall quality of the story.
You should also provide a short explanation for your judgment.

**Story to evaluate:**
{story}

Please provide your assessment in the following format, ensuring each score is an integer between 0 and 100:
{{"EXPLANATION": "The dialogue is coherent, but the phrasing is slightly off.","ORIGINALITY": 0, "COHERENCE": 0, "GRAMMAR": 0, "QUALITY": 0}}
"""

        # Get evaluation from judge model
        result = await judge_model.generate(
            evaluation_prompt, config=GenerateConfig(temperature=0.0)
        )

        # Use regex to find a dictionary with explanation and four scores
        dict_match = re.search(
            r'\{"EXPLANATION":\s*"([^"]+)",\s*"ORIGINALITY":\s*(\d+),\s*"COHERENCE":\s*(\d+),\s*"GRAMMAR":\s*(\d+),\s*"QUALITY":\s*(\d+)\}',
            result.completion,
        )

        if dict_match:
            explanation = dict_match.group(1)
            scores = {
                "originality": int(dict_match.group(2)),
                "coherence": int(dict_match.group(3)),
                "grammar": int(dict_match.group(4)),
                "quality": int(dict_match.group(5)),
            }
        else:
            explanation = ""
            scores = {"originality": 0, "coherence": 0, "grammar": 0, "quality": 0}

        scores = {k: max(0, min(v, 100)) for k, v in scores.items()}

        return Score(
            value=scores,  # type:ignore
            answer=story,
            explanation=explanation,
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
    eval(
        evaluate_story_generation,
        model=model,
        limit=100,
        max_tokens=200,
        max_connections=1,  # Something is seriously broken with the parallelism,
    )
