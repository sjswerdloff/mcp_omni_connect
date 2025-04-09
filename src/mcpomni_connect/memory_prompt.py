EPISODIC_MEMORY_PROMPT = """
You are analyzing conversations to create structured memories that will improve future interactions. Extract key patterns, preferences, and strategies rather than specific content details.

Review the conversation carefully and create a memory reflection following these rules:

1. Use "N/A" for any field with insufficient information
2. Be ruthlessly concise - each field must be exactly one clear, actionable sentence
3. Focus on reusable interaction patterns and communication strategies
4. Context_tags should balance specificity (to match similar situations) and generality (to be reusable)

Output valid JSON in exactly this format:
{
  "context_tags": [              // 2-4 specific but reusable conversation categories
    string,                      // e.g., "technical_troubleshooting", "emotional_support", "creative_collaboration"
    ...
  ],
  "conversation_summary": string, // One sentence describing the core accomplishment or outcome
  "user_intent": string,          // One sentence capturing what the user was trying to achieve
  "user_preferences": string,     // Notable communication style or information preferences observed
  "effective_strategies": string, // Most successful approach that led to positive outcomes
  "friction_points": string,      // What caused confusion or impeded progress in the conversation
  "follow_up_potential": [        // 0-2 likely topics that might arise in future related conversations
    string,
    ...
  ]
}

Examples of EXCELLENT entries:

Context tags:
["system_integration", "error_diagnosis", "technical_documentation"]
["career_planning", "skill_prioritization", "industry_transition"]
["creative_block", "writing_technique", "narrative_structure"]

Conversation summary:
"Diagnosed and resolved authentication failures in the user's API implementation"
"Developed a structured 90-day plan for transitioning from marketing to data science"
"Helped user overcome plot inconsistencies by restructuring their novel's timeline"

User intent:
"User needed to fix recurring API errors without understanding the authentication flow"
"User sought guidance on leveraging existing skills while developing new technical abilities"
"User wanted to resolve contradictions in their story without major rewrites"

User preferences:
"Prefers step-by-step technical explanations with concrete code examples"
"Values practical advice with clear reasoning rather than theoretical frameworks"
"Responds well to visualization techniques and structural metaphors"

Effective strategies:
"Breaking down complex technical concepts using familiar real-world analogies"
"Validating emotional concerns before transitioning to practical solutions"
"Using targeted questions to help user discover their own insight rather than providing direct answers"

Friction points:
"Initial misunderstanding of the user's technical background led to overly complex explanations"
"Providing too many options simultaneously overwhelmed decision-making"
"Focusing on implementation details before establishing clear design requirements"

Follow-up potential:
["Performance optimization techniques for the implemented solution"]
["Interview preparation for technical role transitions"]
["Character development strategies that align with plot structure"]

Do not include any text outside the JSON object in your response.

Here is the conversation to analyze:

{conversation}
"""