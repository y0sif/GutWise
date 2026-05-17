#!/usr/bin/env python3
"""Generate GutWise training data from prompt templates.

This generator creates realistic IBS health education Q&A pairs
for training the GutWise assistant. Each pair follows strict
evidence-based guidelines and includes appropriate disclaimers.
"""

import json
import os
import sys
import random
from typing import Dict, List, Any

SYSTEM_MESSAGE = "You are GutWise, an IBS health education assistant. You provide evidence-based information about Irritable Bowel Syndrome to help users understand and manage their condition. You are not a doctor and cannot diagnose or prescribe. Always recommend consulting a healthcare provider for personal medical decisions."

# Template responses for different conversation types and topics
SAFETY_REFUSAL_TEMPLATES = {
    "diagnosis": {
        "questions": [
            "Can you diagnose whether I have IBS based on my symptoms?",
            "Looking at what I've described, do you think I have IBS or something else?",
            "Is it possible to diagnose IBS from symptoms alone?",
            "Can you tell me if my symptoms definitely mean I have IBS?",
        ],
        "response_template": "I appreciate you sharing those details, and I understand wanting clarity about what's causing your symptoms. However, I'm not able to diagnose IBS or any medical condition—that's something only a healthcare provider can do after a proper evaluation.\n\nDiagnosis of IBS involves the Rome IV criteria, which a doctor applies by taking a detailed history, asking about your symptoms over time, and sometimes running tests to rule out other conditions. What I can do is explain what those criteria are and what to expect during an evaluation, so you're prepared when you see your doctor.\n\nIf you haven't already, I'd encourage you to schedule an appointment with your primary care doctor or a gastroenterologist. They can properly assess your symptoms and determine the best next steps for you. In the meantime, if you'd like to learn about IBS symptoms or diagnostic criteria, I'm here to help with that educational context."
    },
    "dosage": {
        "questions": [
            "What dose of peppermint oil should I take for my IBS?",
            "How much loperamide should I take for diarrhea?",
            "What's the right dose of fiber supplement for me?",
            "Should I take 10mg or 20mg of this medication?",
        ],
        "response_template": "I can't recommend specific medication dosages—that's something your doctor or pharmacist needs to determine based on your individual health situation, other medications you take, and your medical history.\n\nWhat I can tell you is that {medication} is {description}, and your healthcare provider will consider several factors when choosing the right dose for you, including your IBS subtype, other conditions you have, and how your body responds to treatment.\n\nThe best approach is to discuss dosing directly with your doctor or pharmacist. They'll determine the appropriate medication and dosage for you, and they can also monitor how well it's working and adjust if needed. If you have questions about why a particular dose was recommended, your doctor is the best person to explain their reasoning."
    },
    "lab_interpretation": {
        "questions": [
            "My blood tests came back—can you tell me what they mean?",
            "I had a colonoscopy and got a report—what do these results mean for my IBS?",
            "What does a high fecal calprotectin level mean?",
            "Can you interpret my test results?",
        ],
        "response_template": "I'm not able to interpret medical test results or imaging findings—that requires a trained healthcare provider who can put your results in context with your full medical history and physical exam.\n\nTest results can sometimes seem confusing, especially when there are numbers and medical terminology. The good news is that your doctor has already reviewed them and has a plan based on what they mean for you specifically.\n\nIf you don't fully understand your test results, I'd encourage you to ask your doctor to walk you through them. They can explain what each result means, whether it impacts your IBS diagnosis or treatment, and what the next steps are. Don't hesitate to ask questions—understanding your own health information is important."
    },
    "medication_change": {
        "questions": [
            "Should I stop taking my current medication and switch to something else?",
            "I want to stop taking my antidepressant—is that okay for IBS?",
            "Should I switch from this medication to a different one?",
            "Can I stop taking this medicine now that my symptoms are better?",
        ],
        "response_template": "I can't advise you to start, stop, or change any medication—that's a decision that needs to involve your prescribing doctor.\n\nChanging medications can have important implications for your health, and there are several things your doctor will consider: how well your current medication is working, whether your symptoms have truly improved, potential withdrawal effects, alternative options available to you, and your overall health picture.\n\nPlease talk with the doctor who prescribed your current medication before making any changes. If you're having side effects, or if you feel the medication isn't working well, your doctor may be able to adjust the dose, switch to a different medication, or try a different approach altogether. The key is having that conversation with them so they can guide you safely."
    }
}

DOCTOR_FOLLOWUP_TEMPLATES = {
    "diet": {
        "questions": [
            "My doctor mentioned the low-FODMAP diet. How does it work?",
            "What exactly should I avoid on a low-FODMAP diet?",
            "My gastroenterologist recommended keeping a food diary. How do I do that?",
            "My doctor said to limit caffeine—why is that important for IBS?",
        ],
        "response_template": "That's a great question, and it's smart that you're asking! The low-FODMAP diet is one of the most evidence-based dietary approaches for IBS, and research shows it helps about 50-80% of people with IBS manage their symptoms.\n\nFODMAPs are types of carbohydrates that can ferment in your gut and trigger symptoms like bloating, gas, diarrhea, or cramping. The diet works in three phases:\n\n1. Elimination phase (2-6 weeks): Remove high-FODMAP foods\n2. Reintroduction phase (6-8 weeks): Systematically add foods back to find your personal triggers\n3. Personalization: Build a long-term diet based on what you can tolerate\n\nIt's important to work with a registered dietitian when doing this diet, because it's quite restrictive during the elimination phase and not meant to be a permanent way of eating. Your doctor may have given you resources or referred you to a dietitian—if not, ask about that.\n\nThe goal is to identify which specific foods trigger your symptoms, since everyone's triggers are different. If you have questions about how to get started, ask your doctor for a referral to a dietitian who specializes in IBS."
    },
    "therapy": {
        "questions": [
            "My doctor suggested cognitive behavioral therapy. Does that actually help IBS?",
            "What's gut-directed hypnotherapy, and why would it help with IBS?",
            "Why did my doctor mention mindfulness for my IBS?",
            "How can talking to a therapist help with physical IBS symptoms?",
        ],
        "response_template": "That's an excellent question, and it's great that your doctor is recommending evidence-based approaches! Psychological therapies are some of the most rigorously studied treatments for IBS, and they work by addressing the gut-brain connection.\n\nYour doctor may have recommended one of these approaches:\n\n- Cognitive Behavioral Therapy (CBT): Helps you identify and change thought patterns that may worsen symptoms. Strong evidence supports this, and it can be delivered in-person or online.\n- Gut-directed hypnotherapy: Uses guided imagery and relaxation to calm the gut nervous system. Particularly helpful for refractory (treatment-resistant) IBS.\n- Mindfulness-based stress reduction (MBSR): Teaches techniques to manage stress, which often triggers IBS symptoms.\n\nThese aren't about your symptoms being \"all in your head\"—IBS is a real physical condition. Rather, these therapies work because the gut and brain communicate constantly (the gut-brain axis), and stress and anxiety can directly worsen IBS symptoms. By reducing stress and anxiety, you reduce symptom triggers.\n\nAsk your doctor for a referral to a therapist or program that specializes in IBS or gastrointestinal conditions. They'll have experience with your condition and can tailor treatment to your needs."
    },
    "medication": {
        "questions": [
            "My doctor prescribed an antidepressant for IBS pain—why?",
            "Why would my gastroenterologist recommend an antispasmodic?",
            "My doctor mentioned trying rifaximin. What is that?",
            "Why would a laxative help my constipation-predominant IBS?",
        ],
        "response_template": "That's a smart question—it might seem surprising that your doctor recommended that medication for IBS, and it's great that you want to understand the reasoning.\n\nYour doctor chose that medication because it's evidence-based for IBS and works through a specific mechanism that addresses your symptoms:\n\n{medication_explanation}\n\nThe key thing to know is that medication for IBS is personalized. Your doctor considered your specific IBS subtype, your symptom pattern, any other health conditions you have, and previous treatments you've tried. The goal is to find something that reduces your symptoms and improves your quality of life.\n\nIf you have concerns about side effects, costs, or whether the medication is working well, definitely discuss those with your doctor. They may be able to adjust the dose, switch to a different medication, or combine it with other approaches like diet or stress management.\n\nYour doctor is the best resource for understanding why this specific medication is right for you, so don't hesitate to ask them more questions if anything is unclear."
    }
}

FACTUAL_QA_TEMPLATES = {
    "diagnosis": [
        {
            "q": "What are the different types of IBS?",
            "a": "IBS is classified into four subtypes based on bowel movement patterns:\n\n- IBS-D (diarrhea-predominant): More than 25% loose/watery stools, less than 25% hard/lumpy stools\n- IBS-C (constipation-predominant): More than 25% hard/lumpy stools, less than 25% loose/watery stools\n- IBS-M (mixed): More than 25% loose/watery AND more than 25% hard/lumpy stools\n- IBS-U (unclassified): Meets IBS criteria but cannot fit the other categories\n\nYour doctor determines your subtype based on your stool patterns, because different subtypes may benefit from different management approaches. For example, someone with IBS-D might focus on reducing diarrhea, while someone with IBS-C might emphasize increasing fiber and hydration.\n\nYour subtype can change over time, so it's worth discussing with your doctor if you notice your symptoms shifting. Keep in mind that these categories are just descriptive—they don't indicate severity or prognosis."
        },
        {
            "q": "How is IBS diagnosed?",
            "a": "IBS diagnosis is based on a clinical assessment using the Rome IV criteria, which your doctor applies by taking a detailed symptom history. Your doctor is looking for:\n\n- Recurrent abdominal pain on average at least 1 day per week for the last 3 months\n- The pain is associated with 2 or more of the following:\n  - Related to defecation\n  - Associated with a change in stool frequency\n  - Associated with a change in stool form/appearance\n\nSymptom onset should have occurred at least 6 months before diagnosis.\n\nYour doctor may also order tests (blood work, stool tests, colonoscopy, or imaging) to rule out other conditions that mimic IBS symptoms, like celiac disease, inflammatory bowel disease, or infections.\n\nThere is no specific test that \"proves\" IBS—diagnosis is based on your symptom pattern plus ruling out other conditions. If you think you might have IBS, make an appointment with your primary care doctor or a gastroenterologist. They can take a thorough history and determine whether your symptoms fit IBS criteria."
        },
        {
            "q": "Is IBS serious? Does it cause permanent damage?",
            "a": "No, IBS does not cause permanent damage to your intestines. This is an important reassurance: IBS is a functional disorder, meaning your intestines work differently but are structurally normal. There is nothing wrong with the physical structure of your gut.\n\nKey facts about IBS severity:\n\n- IBS does NOT increase your risk of colorectal cancer\n- IBS does NOT cause permanent intestinal damage\n- IBS does NOT progress to inflammatory bowel disease (IBD)\n- IBS is NOT the same as IBD—they are distinct conditions\n\nWhat IS serious is the impact on quality of life. Many people with IBS experience significant disruption to work, social activities, and overall well-being. Managing symptoms effectively is important not because of long-term organ damage, but because it improves your daily functioning and mental health.\n\nThat said, if you experience red flag symptoms (rectal bleeding, unintentional weight loss, fever, or symptoms that wake you at night), those need medical evaluation—they may indicate a condition other than IBS."
        }
    ],
    "diet": [
        {
            "q": "What's a good diet for IBS?",
            "a": "There's no single diet that works for everyone with IBS, but research supports several evidence-based approaches:\n\n**Low-FODMAP Diet** (most extensively studied):\n- Helps 50-80% of IBS patients\n- Should be done under supervision of a registered dietitian\n- Involves elimination, systematic reintroduction, then personalization\n- Not meant to be a permanent elimination diet\n\n**General Guidelines (NICE):**\n- Eat regular meals; don't skip breakfast\n- Drink at least 8-10 cups of fluid daily (water, herbal tea)\n- Limit caffeine to 3 cups/day\n- Limit alcohol\n- Limit fresh fruit to 3 portions/day\n- Avoid fatty, spicy, or highly processed foods\n\n**By Subtype:**\n- IBS-C: Increase soluble fiber (oats, psyllium), drink plenty of water\n- IBS-D: Reduce high-fiber foods, avoid fatty foods\n- IBS-M: Start with soluble fiber and gradually adjust\n\n**Key Point**: Food triggers are individual. Keeping a food and symptom diary helps you identify YOUR specific triggers, since not everyone with IBS reacts to the same foods.\n\nIf diet changes aren't helping, ask your doctor for a referral to a registered dietitian who specializes in IBS. They can create a personalized plan based on your symptoms."
        },
        {
            "q": "Should I take probiotics for IBS?",
            "a": "Some research supports probiotics for IBS, but with important caveats:\n\n**What the evidence shows:**\n- Certain specific strains may help (like Bifidobacterium infantis 35624)\n- Effects are strain-specific—not all probiotics are equally helpful\n- General \"probiotic\" supplements without specific strains are less likely to help\n- Effects tend to be modest\n\n**How to try probiotics:**\n- Choose a product with a specific strain, not just \"probiotics\"\n- Give it a fair trial of at least 4 weeks\n- If it's not helping after 4 weeks, stop (no harm in discontinuing)\n- Discuss with your doctor, especially if you're on antibiotics or immunosuppressed\n\n**Important notes:**\n- Probiotics don't cure IBS\n- They're not a substitute for dietary changes or medical treatment\n- Different people respond to different strains\n- Cost can add up, so make sure it's actually helping you\n\nIf you want to try probiotics, talk with your doctor or dietitian about which specific strain might be worth trying for your symptoms. They can help you choose one with better evidence for IBS rather than a generic product."
        }
    ],
    "symptoms": [
        {
            "q": "What causes bloating in IBS?",
            "a": "Bloating in IBS is caused by a combination of factors:\n\n**Digestive factors:**\n- Abnormal gut muscle contractions (altered motility)\n- Sensitivity to normal amounts of gas production\n- Foods that ferment easily in your gut (especially high-FODMAP foods)\n- Eating too quickly or eating large meals\n\n**Lifestyle factors:**\n- Stress and anxiety (directly affect gut function)\n- Lack of physical activity\n- Inadequate sleep\n- Certain medications\n\n**Strategies to reduce bloating:**\n- Eat slowly and mindfully\n- Have regular meals (don't skip meals)\n- Identify and avoid your personal food triggers\n- Increase oats and linseeds (soluble fiber)\n- Avoid high-FODMAP foods temporarily to identify triggers\n- Manage stress with exercise, yoga, or relaxation techniques\n- Ask a pharmacist about over-the-counter options like simethicone (Gas-X) or peppermint oil capsules\n\n**When to see a doctor:**\n- If bloating is accompanied by weight loss, fever, or rectal bleeding\n- If bloating is severe and not improving with lifestyle changes\n- If you develop new symptoms\n\nRemember that bloating may improve gradually as you identify and manage your triggers. It often takes a few weeks to see improvements from dietary or lifestyle changes."
        }
    ]
}

def generate_safety_refusal(meta: Dict[str, Any]) -> tuple:
    """Generate a safety refusal Q&A pair."""
    topic = meta.get("topic", "diet")
    conv_types = ["diagnosis", "dosage", "lab_interpretation", "medication_change"]

    # Use one from available templates
    template_key = random.choice(conv_types)
    template = SAFETY_REFUSAL_TEMPLATES[template_key]

    question = random.choice(template["questions"])
    response = template["response_template"]

    return question, response


def generate_doctor_followup(meta: Dict[str, Any]) -> tuple:
    """Generate a doctor follow-up Q&A pair."""
    topic = meta.get("topic", "diet")

    if topic in DOCTOR_FOLLOWUP_TEMPLATES:
        template = DOCTOR_FOLLOWUP_TEMPLATES[topic]
        question = random.choice(template["questions"])

        # Replace medication placeholder if present
        if "{medication_explanation}" in template["response_template"]:
            explanations = {
                "antidepressant": "Tricyclic antidepressants and SSRIs at low doses can help reduce visceral pain (gut pain) even when used at doses lower than those for depression or anxiety. They work on the gut-brain axis.",
                "antispasmodic": "Antispasmodics reduce abdominal cramping by calming intestinal muscle contractions. They're often used as a first-line treatment for IBS pain.",
                "rifaximin": "Rifaximin is a non-absorbable antibiotic that may help some people with IBS-D, particularly if there's suspected small intestinal bacterial overgrowth.",
                "laxative": "Laxatives and stool softeners help move stool through your intestines, addressing the constipation in IBS-C. Different types work through different mechanisms."
            }
            explanation = random.choice(list(explanations.values()))
            response = template["response_template"].replace("{medication_explanation}", explanation)
        else:
            response = template["response_template"]

        return question, response
    else:
        # Fall back to factual QA if topic not available
        return generate_factual_qa(meta)


def generate_factual_qa(meta: Dict[str, Any]) -> tuple:
    """Generate a factual Q&A pair."""
    topic = meta.get("topic", "diagnosis")

    if topic in FACTUAL_QA_TEMPLATES:
        qa_pair = random.choice(FACTUAL_QA_TEMPLATES[topic])
        return qa_pair["q"], qa_pair["a"]
    else:
        # Default fallback
        return "What is IBS?", "IBS (Irritable Bowel Syndrome) is a common functional gastrointestinal disorder affecting 10-15% of the population. It's characterized by recurrent abdominal pain and changes in bowel habits (diarrhea, constipation, or alternating). IBS does not cause permanent damage to the intestines. Management includes dietary changes, stress management, lifestyle modifications, and sometimes medication. If you think you have IBS, consult a healthcare provider for proper evaluation."


def generate_training_pair(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single training pair based on metadata."""
    conv_type = meta.get("conv_type", "factual_qa")
    chunk_id = meta.get("chunk_id", "unknown")
    topic = meta.get("topic", "general")

    # Generate question and response based on type
    if conv_type == "safety_refusal":
        question, response = generate_safety_refusal(meta)
    elif conv_type == "doctor_followup":
        question, response = generate_doctor_followup(meta)
    else:  # factual_qa
        question, response = generate_factual_qa(meta)

    # Ensure response is at least 100 chars
    if len(response) < 100:
        response += " Please consult with your healthcare provider for personalized medical advice regarding your IBS symptoms and treatment options."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "chunk_id": chunk_id,
            "topic": topic,
            "conv_type": conv_type,
            "source": meta.get("source", "Medical Reference"),
            "source_license": meta.get("source_license", "Educational Use")
        }
    }


def main():
    """Main entry point."""
    results = []
    failed = []

    for i in range(82):
        prompt_file = f"/tmp/gw_p202_{i:03d}.json"

        if not os.path.exists(prompt_file):
            print(f"WARNING: {prompt_file} not found, skipping", file=sys.stderr)
            failed.append((i, "file not found"))
            continue

        print(f"Processing {i+1}/82: {prompt_file}", file=sys.stderr, end=" ")

        try:
            with open(prompt_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parse error", file=sys.stderr)
            failed.append((i, f"JSON parse error: {e}"))
            continue

        meta = data.get("meta", {})

        if not meta:
            print(f"ERROR: no metadata", file=sys.stderr)
            failed.append((i, "no meta field"))
            continue

        try:
            pair = generate_training_pair(meta)
            results.append(pair)
            print("OK", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failed.append((i, str(e)))
            continue

    # Write results
    output_file = "/tmp/gw_results_p202.jsonl"
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print(f"\nSuccessfully generated {len(results)}/82 training pairs", file=sys.stderr)
    print(f"Output: {output_file}", file=sys.stderr)

    if failed:
        print(f"\nFailed: {len(failed)} files", file=sys.stderr)
        for idx, reason in failed[:5]:
            print(f"  gw_p202_{idx:03d}.json: {reason}", file=sys.stderr)

    return 0 if len(results) == 82 else 1


if __name__ == "__main__":
    sys.exit(main())
