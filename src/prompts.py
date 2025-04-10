# prompt for agent to create a post and react to a feed

class AgentPrompts:
    @staticmethod
    def create_post_prompt(
        persona: str,
        memories_text: str,
        recent_posts_text: str,
        feed_text: str
    ) -> str:
        return f"""Create a social media tweet for a user with the following characteristics:

Background: {persona}

Your recent memories and experiences:
{memories_text if memories_text else "No relevant memories."}

Posts you've made recently:
{recent_posts_text if recent_posts_text else "No recent posts."}

Recent posts by other users in your feed:
{feed_text if feed_text else "No recent feed posts."}

The post should be authentic to the user's persona and background and can reference your past experiences. Keep it concise and suitable for a social media platform.

IMPORTANT: 
- Avoid repeating similar topics or themes from your recent posts
- Try to bring fresh perspectives or discuss different aspects of your interests
- Feel free to engage with or reference one or more recent posts from your feed when relevant
- If there's breaking news in your feed, consider engaging with it if it aligns with your interests, whether you agree or disagree

You don't need to always use emojis every time you write something.

Consider the posts that you have made recently. 
Try to diversify your content and style. For example, avoid always starting a post with the same phrase like "just ..."

The post you are about to create is:
"""

#     @staticmethod
#     def create_feed_reaction_prompt_old(
#         persona: str,
#         memories_text: str,
#         feed_content: str,
#         reflections_text: str = "",
#     ) -> str:
#         return f"""You are browsing your social media feed with the following characteristics:
# Background: {persona}

# Your recent memories and interactions:
# --------------------------------
# {memories_text}
# --------------------------------

# Here are your recent reflections (if any):
# --------------------------------
# {reflections_text}
# --------------------------------

# This is your feed:
# --------------------------------
# {feed_content}
# --------------------------------

# Decide how you want to interact with this feed. You can choose MULTIPLE actions from the following:
#     1. like-post: Like a post you agree with or appreciate
#     2. share-post: Share important/interesting content
#     3. flag-post: Flag harmful/incorrect content
#     4. follow-user: Follow users with aligned interests
#     5. unfollow-user: Unfollow users you no longer want to engage with
#     6. comment-post: Add a brief comment (max 250 chars)
#     7. like-comment: Like insightful comments
#     8. add-note: Add context/fact-checking to posts (max 250 chars)
#     9. rate-note: Rate existing notes. The valid values are "helpful" or "not-helpful"
#     10. ignore: Skip interacting with feed

# Consider sharing news with your network if you find it credible and important.
# And flag the news posts that you think are not credible or harmful.

# Consider the community notes on the posts that you see. These are notes added by other users to the post.
# You can rate the notes as helpful or not-helpful.

# When you think a post is not credible or harmful, you can flag it, and add a community note to it.

# Community Notes Guidelines:
# 1. When you see a post with existing community notes:
#    - First, review the existing notes
#    - If you agree with a note, rate it as "helpful", if you disagree then rate it as "not-helpful"
#    - If you disagree with ALL notes, add your own note
#    - Only add a new note if you have substantially different or additional context to provide
# 2. Add notes when you can provide:
#    - Important missing context
#    - Fact-checking with reliable sources
#    - Clarification of misleading information
#    - Additional relevant information
# 3. Rate existing notes as "helpful" when they:
#    - Provide accurate and verifiable information
#    - Add valuable context
#    - Help readers better understand the post
# 4. Rate notes as "not-helpful" when they:
#    - Contain incorrect information
#    - Are irrelevant or misleading
#    - Show clear bias or propaganda

# Consider your past interactions and reflections when deciding how to engage with the content.
    
# Consider liking/sharing posts that have high quality and align with your interests.

# Choose one or more actions and specify the target, i.e. post/comment/user you want to interact with.

# If you choose to comment or add a note, please provide a short and concise message (max 250 characters).
# For rating notes, specify if it was "helpful" or "not-helpful".

# For each action, provide a brief and concise reasoning for your choice.

# You may also choose to do nothing and just browse the feed. In this case, just specify the action "ignore", and provide a brief and concise reasoning for your choice:
# {{
#     "action": "ignore",
#     "reasoning": "I don't want to interact with this feed"
# }}

# Respond strictly with a JSON object in the following format. Here is an example:
# {{
#     "actions": [
#         {{
#             "action": "<action-on-post>",
#             "target": "<post-id>",
#             "reasoning": "<reasoning>"
#         }},
#         {{
#             "action": "<action-on-user>",
#             "target": "<user-id>",
#             "reasoning": "<reasoning>"
#         }},
#         {{
#             "action": "<comment-post>",
#             "target": "<post-id>",
#             "content": "<comment-content>",
#             "reasoning": "<reasoning>"
#         }},
#         {{
#             "action": "<action-on-comment>",
#             "target": "<comment-id>",
#             "reasoning": "<reasoning>"
#         }},
#         {{
#             "action": "<add-note>",
#             "target": "<post-id>",
#             "content": "<note-content>",
#             "reasoning": "<reasoning>"
#         }},
#         {{
#             "action": "<rate-note>",
#             "target": "<note-id>",
#             "note_rating": "<helpful/not-helpful>",
#             "reasoning": "<reasoning>"
#         }}
#     ]
# }}

# You can choose one or more, but not necessarily all actions for each round.
# """


    @staticmethod
    def create_feed_reaction_prompt_deprecated(
        persona: str,
        memories_text: str,
        feed_content: str,
        reflections_text: str = "",
    ) -> str:
        return f"""You are browsing your social media feed as a user with this background:
{persona}

Recent memories:
{memories_text}

Your feed:
--------------------------------
{feed_content}
--------------------------------

Consider your past interactions and reflections when deciding how to engage with the content.
Pay attention to fact-check verdicts on posts. Posts marked as "false" with high confidence should be treated with caution.

Decide how you want to interact with this feed. You can choose MULTIPLE actions from the following:

1. `like-post`: Like a post you agree with or appreciate
2. `share-post`: Share important/interesting content
3. `flag-post`: Flag harmful/incorrect content

These are the only valid actions you can choose from.

*For each action you choose, give a brief reasoning.*

Respond with a JSON object containing your chosen actions:
{{
    "actions": [
        {{
            "action": "<action-name>",
            "target": "<id-of-post/comment/note/user>",
            "content": "<message-if-needed>",
            "note_rating": "<helpful/not-helpful>",
            "reasoning": "<brief-reason>"
        }}
    ]
}}"""


    def create_feed_reaction_prompt(
        persona: str,
        memories_text: str,
        feed_content: str,
        reflections_text: str = "",
        experiment_type: str = "third_party_fact_checking",
        include_reasoning: bool = False
    ) -> str:
        # Base prompt that's common across all experiment types
        base_prompt = f"""You are browsing your social media feed as a user with this background:
{persona}

Recent memories and interactions:
{memories_text if memories_text else "No relevant memories."}

Your feed:
--------------------------------
{feed_content if feed_content else "No recent feed posts."}
--------------------------------

Your past reflections:
{reflections_text if reflections_text else "N/A"}

Based on your persona, memories, and the content you see, choose how to interact with the feed.
"""
        if not experiment_type:
            raise ValueError("Experiment type is required")

        # Add experiment-specific instructions and valid actions
        if experiment_type == "no_fact_checking":
            base_prompt += """
Valid actions:
- like-post // [post_id]
- share-post // [post_id]
- comment-post // [post_id] with [content], limited to 250 characters
- ignore

Interact with posts and users based on your interests and beliefs. 
If the information seems surprising or novel, feel free to engage with it and share it with your network.
"""
        elif experiment_type == "third_party_fact_checking":
            base_prompt += """
Valid actions:
- like-post // [post_id]
- share-post // [post_id]
- comment-post // [post_id] with [content], limited to 250 characters
- ignore
"""
        elif experiment_type == "community_fact_checking":
            base_prompt += """
You can add community notes to posts that you think need additional context or fact-checking.
You can also rate existing community notes as helpful or not helpful based on their accuracy and usefulness.

Valid actions:
- like-post // [post_id]
- share-post // [post_id]
- comment-post // [post_id] with [content], limited to 250 characters
- add-note // [post_id] with [content] - Add a community note to provide context or fact-checking
- rate-note // [note_id] as [helpful/not-helpful] - Rate existing community notes
- ignore

If you see existing community notes on a post, first consider rating them as helpful or not helpful, and then add your own note ONLY if you have additional context to provide.
"""
        elif experiment_type == "hybrid_fact_checking":
            base_prompt += """
Pay attention to both official fact-check verdicts and community notes on posts.
You can add your own community notes and rate existing ones, while also considering official fact-checks.

Valid actions:
- like-post // [post_id]
- share-post // [post_id]
- comment-post // [post_id] with [content], limited to 250 characters
- add-note [post_id] with [content] - Add a community note to provide context or fact-checking
- rate-note [note_id] as [helpful/not-helpful] - Rate existing community notes
- ignore
"""

        base_prompt += """
THESE ARE THE ONLY VALID ACTIONS YOU CAN CHOOSE FROM.
"""

        # Add reasoning instructions if enabled
        if include_reasoning:
            base_prompt += """
For each action you choose, give a brief reasoning explaining your decision.
"""

        base_prompt += """
Respond with a JSON object containing a list of actions. For each action, include:
- action: The action type from the valid actions list
- target: The ID of the post/user/comment/note (not needed for 'ignore')
- content: Required for comment-post and add-note actions
"""

        # Add reasoning field to example if enabled
        if include_reasoning:
            base_prompt += """
- reasoning: A brief explanation of why you took this action
"""

        # Add note_rating field for relevant experiment types
        if experiment_type in ["community_fact_checking", "hybrid_fact_checking"]:
            base_prompt += """
- note_rating: Required for rate-note actions ("helpful" or "not-helpful")
"""

        # Example response
        if include_reasoning:
            base_prompt += """
Example response:
{
    "actions": [
        {
            "action": "like-post",
            "target": "post-123",
            "reasoning": "This post contains valuable information"
        },
        {
            "action": "share-post",
            "target": "post-123",
            "reasoning": "I want to spread this important news"
        }
    ]
}"""
        else:
            base_prompt += """
Example response:
{
    "actions": [
        {
            "action": "like-post",
            "target": "post-123"
        },
        {
            "action": "share-post",
            "target": "post-123"
        }
    ]
}"""

        return base_prompt


    @staticmethod
    def create_reflection_prompt(persona: dict, memory_text: str) -> str:
        """Create a prompt for generating agent reflections based on recent memories."""
        return f"""Based on your recent experiences as a social media user with:
Background: {persona}

Recent memories and experiences:
{memory_text}

Reflect on these experiences and generate insights about:
1. Patterns in your interactions
2. Changes in your relationships
3. Evolution of your interests
4. Potential biases or preferences you've developed
5. Goals or objectives you might want to pursue

Provide a thoughtful reflection that could guide your future behavior. Do not use bullet points, just summarize into one short and concise paragraph.
"""

    @staticmethod
    def get_system_prompt() -> str:
        """SYSTEM PROMPT FOR AGENT"""
        return f"""You are a social media user browsing on the internet. Use a conversational tone."""

class FactCheckerPrompts:
    @staticmethod
    def get_system_prompt() -> str:
        return """You are an expert fact-checker working to verify social media content. 
        Your role is to:
        1. Analyze claims made in posts
        2. Research and verify factual accuracy
        3. Provide clear, evidence-based verdicts
        4. Cite reliable sources
        5. Maintain objectivity and thoroughness
        
        Your verdicts must be well-researched and carefully considered."""

    def create_fact_check_prompt(
        post_content: str,
        community_notes: str,
        engagement_metrics: dict
    ) -> str:
        return f"""Please fact-check the following social media post:

Content: {post_content}

Engagement Metrics:
- Likes: {engagement_metrics['likes']}
- Shares: {engagement_metrics['shares']}
- Comments: {engagement_metrics['comments']}
{community_notes}

Please analyze this content and provide:
1. A verdict (true/false/unverified) - if you are unsure, mark it as unverified
2. A detailed explanation of your findings
3. Your confidence level (0.0 to 1.0)
4. List of sources consulted

If the post mentions a time that is in the future or has content that is outside of your knowledge scope, you should mark it as unverified.
For obvious misinformation, you should mark it as false.

Format your response as a structured verdict with these components."""