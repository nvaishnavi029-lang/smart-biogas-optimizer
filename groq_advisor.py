import os
from groq import Groq
import streamlit as st

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

@st.cache_data(show_spinner=False)
def cached_ai(prompt):
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500   #  response
    )
    return res.choices[0].message.content

def biogas_ai_advisor(inputs, biogas):

    if client is None:
        return "AI advisor unavailable (set GROQ_API_KEY)"
    prompt = f"""
You are a senior biogas plant engineer and anaerobic digestion expert.

Analyze the system and give PRACTICAL and ACTIONABLE advice.

Inputs:
{inputs}

Biogas Output: {biogas}


Respond in this format:

1. Performance Rating:
- Classify as Low / Moderate / High
- Justify with specific reasons based on inputs and outputs

2. Critical Issues:
- Identify exact problems in mix (composition, imbalance, conditions)

3. Optimization Strategy:
- Give step-by-step strategy to improve gas output
- Mention specific materials to increase/decrease

4. Real-World Recommendations:
- Industrial or farm-level suggestions
- Avoid generic statements

Rules:
- Be specific
- Use numbers where possible
- No vague answers
- Limit to 500 tokens
"""
    try:
        return cached_ai(prompt)
    except Exception as e:
        return str(e)