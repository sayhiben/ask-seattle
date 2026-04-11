# Labeling Policy

Use this policy when creating training and evaluation data. The goal is to model the moderation rule, not personal taste.

## Positive: `askseattle`

Label a submission `askseattle` when the main purpose is a recurring advice, recommendation, planning, or basic information request that the subreddit wants redirected.

Common positive cases:

- Visitor itinerary planning or itinerary review.
- Where to stay, what neighborhood is safe, or how to get around as a visitor.
- Moving to Seattle, moving away from Seattle, neighborhood selection, commute tradeoffs, and cost-of-living questions.
- Product, restaurant, bar, tattoo, medical, dental, vet, pet, or service recommendations.
- Basic city information that can be answered by existing guides, search, or official resources.
- Vacation advice, day trip advice, and airport or transit planning.
- Legal, landlord, lease, ticket, or employment advice when the post is primarily asking the subreddit what to do.

## Negative: `not_askseattle`

Label a submission `not_askseattle` when it is not a recurring advice or recommendation request, even if it contains a question mark.

Common negative cases:

- Local news, politics, policy discussion, or public agency updates.
- Transit alerts, weather alerts, safety incidents, power outages, and road closures.
- Lost and found, missing pets, community announcements, volunteer events, AMAs, and moderation posts.
- Original discussion prompts about Seattle issues.
- Photos, local observations, trip reports, or follow-up reports that are not asking for planning advice.
- Narrow factual questions that are clearly tied to a current local event and not reusable advice content.

## Borderline Cases

Use `askseattle` when the title and body are mostly a request for personalized recommendations or planning help, even if the post mentions a specific date, budget, or personal context.

Use `not_askseattle` when the post does not clearly fit the redirect category. Borderline cases should be resolved as binary labels during review rather than stored as a third class.

## Evaluation Sets

Keep a held-out evaluation set from real subreddit history. Do not repeatedly hand-tune on the same examples.

For automatic removals, track precision in the `auto` band. Recall matters, but a high false positive rate is more damaging than missing some low-value posts.
