# ==========================================================
# MARKSHEET AI AGENT
# Gemini Vision + Interactions API
# ==========================================================

import os
import json
import base64
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
from google import genai


# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing in .env"
    )


# ==========================================================
# GEMINI CLIENT
# ==========================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={
        "timeout": 180000
    }
)


# ==========================================================
# MARKSHEET PROMPT
# ==========================================================

MARKSHEET_PROMPT = """
You are an AI Academic Marksheet Analyzer.

Analyze the uploaded marksheet image carefully.

Your job is to extract academic information from the image.

IMPORTANT RULES:

1. Do NOT guess any value.
2. Extract only information that is clearly visible.
3. If a value is unclear or unavailable, use null.
4. Identify whether the marksheet is:
   - 10th
   - 12th
   - College
   - Unknown

5. Extract:
   - Student name
   - Register number
   - School or college
   - Exam year
   - Semester
   - Subjects
   - Marks
   - Maximum marks
   - Grade
   - Grade point
   - Credits
   - Total marks
   - Maximum total marks

6. For 10th and 12th marksheets:
   Extract the subject marks and total marks.

7. For college marksheets:
   Extract subject, marks, grade, grade point and credits
   whenever visible.

8. Do NOT calculate SGPA or CGPA unless enough
   information is clearly available.

9. Return ONLY valid JSON.

Use exactly this structure:

{
    "marksheet_type": null,
    "student_name": null,
    "register_number": null,
    "school_or_college": null,
    "exam_year": null,
    "semester": null,
    "subjects": [
        {
            "subject": null,
            "marks": null,
            "max_marks": null,
            "grade": null,
            "grade_point": null,
            "credits": null
        }
    ],
    "total_marks": null,
    "maximum_total_marks": null,
    "percentage": null,
    "remarks": null
}
"""


# ==========================================================
# IMAGE OPTIMIZATION
# ==========================================================

def optimize_image(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    )

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Reduce image size
    max_dimension = 1200

    width, height = image.size

    if max(width, height) > max_dimension:

        scale = (
            max_dimension /
            max(width, height)
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.LANCZOS
        )

    output = BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=75,
        optimize=True
    )

    return output.getvalue()


# ==========================================================
# EXTRACT MARKSHEET
# ==========================================================

def extract_marksheet(
    image_bytes,
    mime_type="image/jpeg"
):

    if not image_bytes:

        return {
            "error": "No image received.",
            "error_type": "InvalidImage"
        }

    # ------------------------------------------------------
    # IMAGE OPTIMIZATION
    # ------------------------------------------------------

    try:

        print("======================================")
        print("📸 Optimizing marksheet image...")
        print("======================================")

        optimized_image = optimize_image(
            image_bytes
        )

        print(
            f"Original size: "
            f"{len(image_bytes) / 1024:.1f} KB"
        )

        print(
            f"Optimized size: "
            f"{len(optimized_image) / 1024:.1f} KB"
        )

    except Exception as error:

        return {
            "error": "Could not process image.",
            "error_type": type(error).__name__,
            "details": str(error)
        }


    # ------------------------------------------------------
    # CONVERT IMAGE TO BASE64
    # ------------------------------------------------------

    try:

        image_base64 = base64.b64encode(
            optimized_image
        ).decode("utf-8")

    except Exception as error:

        return {
            "error": "Could not encode image.",
            "error_type": type(error).__name__,
            "details": str(error)
        }


    # ------------------------------------------------------
    # SEND TO GEMINI
    # ------------------------------------------------------

    try:

        print("======================================")
        print("🤖 Sending image to Gemini...")
        print("🤖 Model: gemini-3.7-flash")
        print("⏱️ Timeout: 180 seconds")
        print("======================================")

        interaction = client.interactions.create(

            model="gemini-3.7-flash",

            input=[
                {
                    "type": "text",
                    "text": MARKSHEET_PROMPT
                },
                {
                    "type": "image",
                    "data": image_base64,
                    "mime_type": "image/jpeg"
                }
            ]
        )


        # --------------------------------------------------
        # GET RESPONSE
        # --------------------------------------------------

        result = interaction.output_text

        if not result:

            return {
                "error": "Gemini returned an empty response.",
                "error_type": "EmptyResponse"
            }


        print("======================================")
        print("✅ Gemini response received")
        print("======================================")


        # --------------------------------------------------
        # CLEAN JSON
        # --------------------------------------------------

        result = result.strip()

        # Remove markdown JSON fences if present
        if result.startswith("```json"):

            result = result[7:]

        elif result.startswith("```"):

            result = result[3:]


        if result.endswith("```"):

            result = result[:-3]


        result = result.strip()


        # --------------------------------------------------
        # PARSE JSON
        # --------------------------------------------------

        try:

            data = json.loads(result)

        except json.JSONDecodeError as error:

            return {
                "error": "Gemini returned invalid JSON.",
                "error_type": "JSONDecodeError",
                "details": str(error),
                "raw_response": result
            }


        print("======================================")
        print("✅ Marksheet extraction successful")
        print("======================================")


        return data


    # ------------------------------------------------------
    # GEMINI ERROR
    # ------------------------------------------------------

    except Exception as error:

        error_type = type(error).__name__
        error_text = str(error)

        print("======================================")
        print("❌ GEMINI ERROR")
        print("======================================")
        print("Error Type:", error_type)
        print("Error:", error_text)

        return {
            "error": "Gemini API request failed.",
            "error_type": error_type,
            "details": error_text
        }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("======================================")
    print("📄 MARKSHEET AI AGENT")
    print("======================================")
    print("🤖 Gemini Vision")
    print("🤖 Interactions API")
    print("⏱️ Timeout: 180 seconds")
    print("🖼️ Image optimization: Enabled")
    print("======================================")
    print("MARKSHEET AGENT READY")