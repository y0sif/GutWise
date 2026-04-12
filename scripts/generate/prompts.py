"""Prompt templates for generating IBS health education Q&A training data.

Each conversation type has a builder function that takes chunk_text and
reference_context and returns the full prompt string to send to Claude.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt for the GENERATOR (sent to Claude when creating training data)
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = """\
You are generating training data for GutWise, an IBS health education assistant.
Generate a realistic patient question and a helpful, accurate educational response
about IBS (Irritable Bowel Syndrome).

RULES:
- ONLY use information from the provided source material and reference facts
- Include appropriate disclaimers about consulting healthcare providers
- Use empathetic, patient-friendly language
- NEVER diagnose or prescribe specific dosages
- If the topic involves medications, mention them by class and note a doctor should guide choices
- If asked about red flag symptoms, always recommend seeking medical attention
- Frame all information as educational: "research suggests", "guidelines recommend", \
"many patients find"
"""

# ---------------------------------------------------------------------------
# System prompt embedded in the GENERATED training data (ChatML)
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT = (
    "You are GutWise, an IBS health education assistant. You provide evidence-based "
    "information about Irritable Bowel Syndrome to help users understand and manage "
    "their condition. You are not a doctor and cannot diagnose or prescribe. Always "
    "recommend consulting a healthcare provider for personal medical decisions."
)

# ---------------------------------------------------------------------------
# Conversation-type weights (must sum to 1.0)
# ---------------------------------------------------------------------------

CONV_TYPE_WEIGHTS: dict[str, float] = {
    "factual_qa": 0.40,
    "anxious_patient": 0.15,
    "doctor_followup": 0.10,
    "myth_busting": 0.10,
    "safety_refusal": 0.15,
    "multi_turn": 0.10,
}

# ---------------------------------------------------------------------------
# Shared preamble injected into every prompt
# ---------------------------------------------------------------------------

_REFERENCE_BLOCK = """\
=== VERIFIED MEDICAL REFERENCE ===
{reference_context}
=== END REFERENCE ===

=== SOURCE CHUNK (use this as the basis for the question topic) ===
{chunk_text}
=== END SOURCE CHUNK ===
"""

# ---------------------------------------------------------------------------
# Output format instructions
# ---------------------------------------------------------------------------

_SINGLE_TURN_FORMAT = """\
Output EXACTLY in this format (no extra text outside the tags):

<question>
{the patient's question}
</question>

<response>
{GutWise's educational response}
</response>
"""

_MULTI_TURN_FORMAT = """\
Output a 2-3 turn conversation EXACTLY in this format (no extra text outside the tags):

<turn1_question>
{the patient's initial question}
</turn1_question>

<turn1_response>
{GutWise's educational response}
</turn1_response>

<turn2_question>
{a natural follow-up question from the patient}
</turn2_question>

<turn2_response>
{GutWise's follow-up response}
</turn2_response>

You may optionally add a third turn (turn3_question / turn3_response) if it flows naturally.
"""


# ---------------------------------------------------------------------------
# Template builders — each returns the full user prompt for Claude
# ---------------------------------------------------------------------------


def factual_qa(chunk_text: str, reference_context: str) -> str:
    """Simple factual question and educational answer (40% of data)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate a FACTUAL Q&A pair about IBS.

The patient asks a clear, direct question about an IBS topic covered in the source chunk.
The response should be informative, well-structured, and grounded in the reference material.
Use bullet points or numbered lists where appropriate.
Include a brief disclaimer about consulting a healthcare provider.

Examples of the style of question:
- "What are the different types of IBS?"
- "How is IBS diagnosed?"
- "What foods should I avoid with IBS?"

{_SINGLE_TURN_FORMAT}
"""


def anxious_patient(chunk_text: str, reference_context: str) -> str:
    """Worried/scared patient tone, model responds with empathy + facts (15%)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate an ANXIOUS PATIENT Q&A pair about IBS.

The patient writes in a worried, scared, or emotionally distressed tone about their
IBS-related concern. They may use phrases like "I'm really scared", "I'm worried",
"I can't stop thinking about", "Is this normal?", etc.

GutWise's response should:
1. Acknowledge their feelings with empathy ("I understand this is worrying...")
2. Provide reassuring, evidence-based information from the reference material
3. Clearly distinguish when symptoms are typical IBS vs. when they should see a doctor
4. End with encouragement and a gentle recommendation to discuss concerns with their provider

Examples of the style of question:
- "I'm really scared that my stomach pain might be something serious..."
- "I've been having diarrhea for weeks and I'm terrified it could be cancer..."
- "My symptoms are getting worse and I don't know what to do anymore..."

{_SINGLE_TURN_FORMAT}
"""


def doctor_followup(chunk_text: str, reference_context: str) -> str:
    """Patient asking about something their doctor mentioned (10%)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate a DOCTOR FOLLOW-UP Q&A pair about IBS.

The patient is asking about something their doctor, gastroenterologist, or dietitian
recently mentioned or recommended. They want to understand it better.

The question should reference a real medical recommendation (from the source chunk) and
frame it as "My doctor said..." or "My gastroenterologist recommended..." or similar.

GutWise's response should:
1. Explain the concept/treatment the doctor mentioned in patient-friendly language
2. Validate that it's a recognized, evidence-based approach
3. Provide practical details about what to expect
4. Reinforce that following their doctor's guidance is the right approach
5. Offer to explain any specific aspect in more detail

Examples of the style of question:
- "My doctor mentioned I should try a low-FODMAP diet. What exactly is that?"
- "My gastroenterologist wants me to try gut-directed hypnotherapy. Does that actually work?"
- "My doctor prescribed a low-dose antidepressant for my IBS pain. Why?"

{_SINGLE_TURN_FORMAT}
"""


def myth_busting(chunk_text: str, reference_context: str) -> str:
    """'Is it true that...' questions, model corrects misconceptions (10%)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate a MYTH-BUSTING Q&A pair about IBS.

The patient asks about a common misconception, myth, or piece of misinformation about IBS.
They may have read it online, heard it from a friend, or assumed it based on common beliefs.

Use the HALLUCINATION BLOCKLIST in the reference to identify real myths to bust.

GutWise's response should:
1. Clearly but gently correct the misconception
2. Explain what the evidence actually shows
3. Cite the type of evidence (e.g., "according to current gastroenterology guidelines...")
4. Acknowledge why the myth might seem plausible
5. Provide the accurate information as a replacement

Examples of the style of question:
- "Is it true that IBS can turn into cancer?"
- "I heard IBS is just caused by stress and is all in your head. Is that true?"
- "Someone told me I should do a colon cleanse to cure my IBS. Would that help?"
- "Is it true that everyone with IBS should go gluten-free?"

{_SINGLE_TURN_FORMAT}
"""


def safety_refusal(chunk_text: str, reference_context: str) -> str:
    """Questions where the model must decline and redirect (15%)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate a SAFETY REFUSAL Q&A pair about IBS.

The patient asks a question that GutWise should NOT fully answer because it would involve:
- Diagnosing a condition
- Recommending specific medication dosages
- Interpreting lab results or test findings
- Advising on stopping or changing prescribed medications
- Making claims about conditions beyond IBS scope

GutWise's response should:
1. Acknowledge the question warmly without dismissing the patient
2. Clearly explain WHY it cannot provide this specific guidance
3. Explain what a healthcare provider would consider when making this decision
4. Provide general educational context that IS within scope
5. Strongly recommend consulting their doctor for personalized advice

Generate ONE of these types (vary across the dataset):
- Diagnosis request: "Can you diagnose whether I have IBS based on my symptoms?"
- Dosage request: "What dose of [medication from chunk] should I take for my IBS?"
- Lab interpretation: "My doctor ran blood tests — can you tell me what they mean for my IBS?"
- Medication change: "Should I stop taking [medication] and switch to something else?"
- Out-of-scope condition: "I also have [other condition] — how does that interact with my IBS?"

The question topic should relate to the source chunk content where possible.

{_SINGLE_TURN_FORMAT}
"""


def multi_turn(chunk_text: str, reference_context: str) -> str:
    """2-3 turn conversations with follow-up questions (10%)."""
    return f"""\
{_REFERENCE_BLOCK.format(chunk_text=chunk_text, reference_context=reference_context)}

Generate a MULTI-TURN conversation about IBS (2-3 turns).

Create a natural conversation where the patient asks an initial question, receives
an educational response, then asks a follow-up question that builds on or deepens
the topic. The follow-up should feel natural, like something a real patient would ask
after receiving the first response.

Guidelines:
- Turn 1: Patient asks a question about the topic in the source chunk
- Turn 1 response: GutWise provides a thorough but not exhaustive answer
- Turn 2: Patient asks a natural follow-up (more detail, practical application, concern)
- Turn 2 response: GutWise goes deeper or addresses the specific follow-up
- Optional Turn 3: Patient asks for clarification or a related practical question
- All responses must stay grounded in the reference material
- Include appropriate disclaimers (at least once in the conversation)

{_MULTI_TURN_FORMAT}
"""


# ---------------------------------------------------------------------------
# Registry: conv_type string -> builder function
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: dict[str, callable] = {
    "factual_qa": factual_qa,
    "anxious_patient": anxious_patient,
    "doctor_followup": doctor_followup,
    "myth_busting": myth_busting,
    "safety_refusal": safety_refusal,
    "multi_turn": multi_turn,
}


def build_prompt(conv_type: str, chunk_text: str, reference_context: str) -> str:
    """Build a generation prompt for the given conversation type.

    Raises KeyError if conv_type is not a valid template name.
    """
    return TEMPLATE_REGISTRY[conv_type](chunk_text, reference_context)
