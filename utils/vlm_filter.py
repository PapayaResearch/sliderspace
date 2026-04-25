import base64
from io import BytesIO
from PIL import Image


DEFAULT_VLM_PROMPT = (
    "Look at this image carefully. Is it a realistic, good quality photograph "
    "of a person's face with no major artifacts, distortions, or deformities?"
)

_APPROVAL_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Submit the approval decision for the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether the image meets quality requirements."
                }
            },
            "required": ["approved"]
        }
    }
}]


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def approve_image(image: Image.Image, model: str, prompt: str = DEFAULT_VLM_PROMPT) -> bool:
    import litellm, json
    response = litellm.completion(
        model=model,
        tools=_APPROVAL_TOOL,
        tool_choice={"type": "function", "function": {"name": "submit_decision"}},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_to_base64(image)}"}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )
    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)
    return result["approved"] == "yes"
