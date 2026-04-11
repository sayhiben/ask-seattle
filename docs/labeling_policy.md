# Labeling Policy

Use this page when assigning `askseattle` or `not_askseattle` labels to reviewed posts.

The goal is to model the moderation rule, not personal taste.

## Positive: `askseattle`

Label a post `askseattle` when its main purpose is asking the community for recurring advice, recommendations, planning help, or basic reusable city guidance.

Common positive cases:

- visitor itineraries or itinerary review
- where to stay, what neighborhood is safe, or how to get around as a visitor
- moving to Seattle, moving away, neighborhood choice, or commute tradeoffs
- product, restaurant, bar, tattoo, medical, dental, vet, pet, or service recommendations
- basic city information that is broadly reusable and better handled by existing guides or repeated recommendations
- vacation planning, day-trip planning, airport planning, or transit planning
- legal, landlord, lease, ticket, or employment advice when the post is primarily asking what to do

## Negative: `not_askseattle`

Label a post `not_askseattle` when it does not primarily belong to that redirect-style advice bucket, even if it contains a question mark.

Common negative cases:

- local news, politics, policy discussion, or civic updates
- transit alerts, weather alerts, safety incidents, power outages, and road closures
- lost and found, missing pets, community announcements, volunteer events, AMAs, and moderation posts
- original discussion prompts about Seattle issues
- photos, local observations, trip reports, or follow-up reports that are not asking for planning help
- narrow factual questions tied to a current event rather than reusable recommendation content

## Borderline Cases

Resolve borderline posts as binary labels during review.

Use `askseattle` when the post is mostly a request for personalized recommendations or planning help, even if it includes specific dates, budget details, or personal context.

Use `not_askseattle` when the redirect category is not clearly the main point of the post.

## Review Tips

- Judge the primary intent, not isolated keywords.
- Use the title and body together.
- A short or empty body does not make a post automatically negative.
- A link or image post can still be `askseattle` if the title is clearly a recommendation request.
- If you change your mind on a post, re-label it. The local training file is last-write-wins by identity.

## Dataset Hygiene

- Keep using real reviewed subreddit examples.
- Avoid repeatedly hand-tuning on the same held-out examples.
- If you later evaluate moderation actions, prefer precision-first analysis on the high-confidence band.

Next:

- [How to label posts](how-to/label-posts.md)
- [Model and thresholds](explanation/model-and-thresholds.md)
