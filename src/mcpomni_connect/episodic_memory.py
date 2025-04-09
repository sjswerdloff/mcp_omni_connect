EPISODIC_MEMORY_PROMPT = """
You are analyzing conversations to create structured memories that will improve future interactions. Extract key patterns, preferences, and strategies rather than specific content details.

Review the conversation carefully and create a memory reflection following these rules:

1. Use "N/A" for any field with insufficient information
2. Be concise but thorough - use up to 3 sentences for complex fields
3. For long conversations, include the most significant elements rather than trying to be comprehensive
4. Context_tags should balance specificity (to match similar situations) and generality (to be reusable)
5. IMPORTANT: Ensure your output is properly formatted JSON with no leading whitespace or text outside the JSON object

Output valid JSON in exactly this format:
{{
  "context_tags": [              // 2-4 specific but reusable conversation categories
    string,                      // e.g., "technical_troubleshooting", "emotional_support", "creative_collaboration"
    ...
  ],
  "conversation_complexity": integer, // 1=simple, 2=moderate, 3=complex multipart conversation
  "conversation_summary": string, // Up to 3 sentences for complex conversations
  "key_topics": [
    string, // List of 2-5 specific topics discussed
    ...
  ],
  "user_intent": string, // Up to 2 sentences, including evolution of intent if it changed
  "user_preferences": string, // Up to 2 sentences capturing style and content preferences
  "notable_quotes": [
    string, // 0-2 direct quotes that reveal important user perspectives
    ...
  ],
  "effective_strategies": string, // Most successful approach that led to positive outcomes
  "friction_points": string,      // What caused confusion or impeded progress in the conversation
  "follow_up_potential": [        // 0-3 likely topics that might arise in future related conversations
    string,
    ...
  ]
}}

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

Notable quotes:
"give me general knowledge about it",
"ok deep dive in the power levels"
"what is the best way to learn about it"

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

"""