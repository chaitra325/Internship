# openai_advisor.py
import os
from openai import OpenAI

SYSTEM_PROMPT = """
You are a course design advisor for online learning platforms like Udemy.
Given course metadata and a model's predicted success (0=low,1=high),
suggest concrete, actionable improvements to increase enrollments and rating.
Be specific about price, duration, difficulty alignment, content focus,
target audience, and marketing/positioning. Respond in concise bullet points.
"""

# Initialize OpenAI client from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_course_suggestions(raw_dict, label, proba):
    """
    Generate course improvement suggestions using OpenAI API.
    Falls back to predefined text based on success prediction if API fails.
    
    Args:
        raw_dict: Dictionary containing course metadata
        label: Predicted label (1 = high success, 0 = low success)
        proba: Probability of high success
    
    Returns:
        String containing suggestions or fallback text
    """
    
    # Use OpenAI if API key is present
    if OPENAI_API_KEY:
        user_prompt = f"""
Course details:
- Title: {raw_dict.get('title', 'N/A')}
- Category: {raw_dict.get('category', 'N/A')}
- Difficulty: {raw_dict.get('difficulty', 'N/A')}
- Price: ₹{raw_dict.get('price', 0)}
- Duration: {raw_dict.get('duration', 0)} hours
- Lectures: {raw_dict.get('lecture_numbers', 0)}
- Current average rating: {raw_dict.get('rating', 0)}
- Reviews: {raw_dict.get('reviews', 0)}

Model prediction:
- Predicted success label: {label} (1 = high, 0 = low)
- Predicted probability of high success: {proba:.2f}

Task:
1. Briefly explain (1–2 lines) what this prediction implies.
2. Give 5–7 concrete suggestions to improve enrollment and rating:
   - Pricing.
   - Duration and structure.
   - Difficulty vs audience.
   - Content (projects, case studies, hands-on).
   - Marketing/positioning.
3. Use short bullet points only.
"""
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            pass

    # Fallback text based on success prediction
    if label == 1:  # High success prediction
        return """
 Great News - Your Course is Predicted to Succeed!

Your course has strong potential to attract students and get great reviews.

# Marketing & Visibility
Tell the right people about your course! Use ads and social media to reach students interested in your topic. Your course is ready for the spotlight!

# Pricing Strategy
Your course metrics look good. You can confidently price it fairly, or offer an introductory discount to get your first students rolling in.

# Course Organization
Break your course into 4-6 hour modules. Students learn better when topics are bite-sized and clear. Think of each module as a complete mini-course.

# Hands-On Learning
Include 2-3 practical projects students can actually use. This makes your course stand out and students feel they're learning real skills they can apply immediately.

# Extra Materials
Provide downloadable templates, checklists, or code samples. Students love having materials they can reference later and use in their own work.

# Student Community
Be active in answering questions and encourage students to share their progress. Happy students become loyal students and give better reviews.

# Growth Through Reviews
When students finish , ask them to leave reviews. More reviews = better visibility = more enrollments!
"""
    else:  # Low success prediction
        return """
 Here's How to Improve Your Course's Success Rate

Your course shows some challenges, but don't worry - these tips can help turn things around!

# Start with Price
Try lowering your price or offering a free preview section. Sometimes the barrier to entry is the biggest hurdle. Let students see what they're getting first!

# Simplify the Learning Path
Shorten your course and break it into smaller, easier-to-digest lessons (1.5-2 hours each). Students prefer quick wins and feel less overwhelmed.

# Know Your Audience
Don't aim for advanced students first. Focus on beginners and intermediate learners. Create a clear, step-by-step learning path they can follow easily.

# Show Real-World Value
Add hands-on projects and real-world examples students can apply right away. Show them HOW the course helps them solve actual problems.

# Write Clear Descriptions
Rewrite your title and description to clearly state who the course is for and what problem it solves. Be specific about benefits, not just features!

# Build Social Proof Fast
Offer a discount or free access to your first batch of students. This helps you generate those crucial early reviews and ratings quickly.

# Create Compelling Promos
Make videos or content samples showing what students will learn. Focus on outcomes and real success stories from actual students.
"""
