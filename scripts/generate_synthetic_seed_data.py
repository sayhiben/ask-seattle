from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path("data/seed/askseattle_synthetic.jsonl")

POSITIVE_EXAMPLES = [
    (
        "Where should we stay for a long weekend in Seattle?",
        "Two adults visiting in May want a walkable hotel area near restaurants and transit.",
    ),
    (
        "Four day itinerary check for first time visitors",
        "Does Pike Place, the waterfront, Fremont, Ballard, and a ferry day make sense without a car?",
    ),
    (
        "Moving from Chicago, what neighborhood fits a remote worker?",
        "Budget is 2600 and I want coffee shops, groceries, and light rail access if possible.",
    ),
    (
        "Best place to find a waterproof jacket today?",
        "I forgot mine and need a store near downtown or Capitol Hill before sightseeing tomorrow.",
    ),
    (
        "Seafood restaurant for parents visiting from out of state",
        "Looking for a classic Seattle dinner that is not impossible to reserve this Saturday.",
    ),
    (
        "Is it safe to stay near Pioneer Square as a tourist?",
        "My hotel is already booked and recent reviews made me nervous about walking back at night.",
    ),
    (
        "Need landlord advice for a broken heater",
        "The apartment has been cold for a week and I am trying to understand Seattle tenant options.",
    ),
    (
        "Can we visit Rainier without renting a car?",
        "We are flying in for vacation and want a day trip from Seattle with public transportation.",
    ),
    (
        "Dog friendly breweries around Ballard?",
        "Looking for places where a medium dog can sit outside while friends visit this weekend.",
    ),
    (
        "Where should I live if I work in South Lake Union?",
        "Comparing Queen Anne, Capitol Hill, and Fremont with a commute under thirty minutes.",
    ),
    (
        "Fine line tattoo artist recommendations",
        "Hoping to find a Seattle artist with books open for a small memorial piece.",
    ),
    (
        "How bad is the gray weather really?",
        "Thinking about moving to Seattle and wondering if the winter gloom is exaggerated.",
    ),
    (
        "Best neighborhood for a car-free student",
        "I will be near UW and want rent, buses, groceries, and safety advice.",
    ),
    (
        "Where can I buy local smoked salmon to take home?",
        "Visitor here looking for a shop that will pack something for a flight.",
    ),
    (
        "Romantic dinner ideas near the Space Needle",
        "Anniversary trip next month and I want something nice but not too formal.",
    ),
    (
        "How early should I get to SeaTac on a Friday morning?",
        "Flight leaves at 8 am and I have heard the security lines can be terrible.",
    ),
    (
        "Best coffee shops to work from in Capitol Hill",
        "Need wifi and a place that will not mind me using a laptop for a few hours.",
    ),
    (
        "Is Ballard a good place to live with a toddler?",
        "We are moving with a young kid and want parks, daycare, and commute advice.",
    ),
    (
        "Where can I find vegan pastries in Seattle?",
        "Looking for bakery recommendations for a friend visiting this weekend.",
    ),
    (
        "One night in Seattle after a cruise",
        "Where should we stay and what should we do if we only have Saturday afternoon and evening?",
    ),
    (
        "Best way to get from SeaTac to West Seattle late",
        "Landing after midnight and trying to decide between transit, rideshare, or a taxi.",
    ),
    (
        "Neighborhood advice for moving near light rail",
        "I work downtown and want to avoid driving most days. What areas should I search?",
    ),
    (
        "Where should I take kids on a rainy day?",
        "Family visiting with two children and we need indoor activities around Seattle.",
    ),
    (
        "Is the Seattle CityPASS worth it?",
        "Tourist planning Space Needle, aquarium, and museums and wondering if the pass saves money.",
    ),
    (
        "Best whale watching tour from Seattle?",
        "Visiting in July and trying to pick a reputable company for a day trip.",
    ),
    (
        "Where can I donate furniture before moving?",
        "Need recommendations for organizations that pick up a couch and table in Seattle.",
    ),
    (
        "Recommendation for a tenant lawyer",
        "My lease has a confusing fee and I want someone local to review it.",
    ),
    (
        "Which area should tourists avoid at night?",
        "Planning a first visit with my parents and trying to choose between hotels.",
    ),
    (
        "Best thrift stores for winter coats",
        "New to Seattle and trying to find something warm and waterproof on a budget.",
    ),
    (
        "Help choosing between Fremont and Wallingford",
        "Moving next month and want neighborhood advice for restaurants, buses, and noise.",
    ),
    (
        "Can I see Olympic National Park as a day trip?",
        "Visitor without a car wondering if this is realistic from Seattle.",
    ),
    (
        "Where should I get my dog groomed?",
        "Looking for recommendations for a nervous older dog near North Seattle.",
    ),
    (
        "Best sushi for a birthday dinner",
        "Seeking Seattle recommendations for a group of five and a moderate budget.",
    ),
    (
        "How expensive are utilities in Seattle apartments?",
        "Moving from Arizona and trying to estimate monthly costs beyond rent.",
    ),
    (
        "Where to stay before an early cruise departure",
        "Need hotel neighborhood advice for a family leaving from the pier in the morning.",
    ),
    (
        "Looking for a good mechanic near Beacon Hill",
        "New to the area and need recommendations for a reliable shop.",
    ),
    (
        "Best scenic ferry ride for visitors",
        "We have half a day free and want a simple ferry experience from downtown.",
    ),
    (
        "Can I live in Seattle on this salary?",
        "Offer is 85000 and I am trying to understand rent, taxes, transit, and groceries.",
    ),
    (
        "Where can I find Korean skincare products?",
        "Visitor looking for shops in Seattle or Bellevue with a decent selection.",
    ),
    (
        "Restaurant recommendations near Climate Pledge Arena",
        "Going to a concert and want dinner within walking distance before the show.",
    ),
    (
        "Is renting in First Hill a mistake?",
        "Apartment looks good but I do not know the neighborhood and want local opinions.",
    ),
    (
        "Best easy hikes without a car",
        "Visiting Seattle and hoping for transit-accessible nature options.",
    ),
    (
        "Where can I buy a used bike?",
        "Moving to Seattle and looking for reliable shops or community bike sales.",
    ),
    (
        "Advice on parking near Pike Place",
        "Taking relatives downtown and wondering where to park without paying too much.",
    ),
    (
        "Best neighborhood for nightlife but not too loud",
        "Moving soon and trying to compare Capitol Hill, Belltown, and Ballard.",
    ),
    (
        "Where can I get a same day passport photo?",
        "Need a Seattle shop recommendation before an appointment tomorrow.",
    ),
    (
        "Food recommendations for a layover",
        "We have six hours near the airport and want something better than terminal food.",
    ),
    (
        "Is public transit enough for a weekend trip?",
        "Visitors staying downtown and hoping to avoid renting a car.",
    ),
    (
        "Best area to stay with elderly parents",
        "Need hotels near attractions but not too much walking or late night noise.",
    ),
    (
        "Where should I take someone who has never seen Seattle?",
        "Friend is visiting for one day and wants the classic local highlights.",
    ),
    (
        "How do I handle a parking ticket?",
        "Rental car got a Seattle ticket and I am not sure whether contesting it is worth it.",
    ),
    (
        "Looking for a cat sitter recommendation",
        "Need someone reliable in Queen Anne while I am away for a week.",
    ),
    (
        "Best cheap eats around UW",
        "Student visiting campus and looking for lunch spots that are not expensive.",
    ),
    (
        "Is a day trip to Vancouver realistic?",
        "Tourists staying in Seattle and wondering if border traffic makes this a bad idea.",
    ),
    (
        "Where can I find gluten free donuts?",
        "Trying to find a Seattle bakery recommendation for a friend with celiac.",
    ),
    (
        "Advice for moving out of Seattle",
        "Considering Tacoma or Portland and wondering how people compare the cost and commute.",
    ),
    (
        "Best place to watch the sunset",
        "Visitor wants a scenic viewpoint that is easy to reach from downtown.",
    ),
    (
        "Where to buy last minute camping gear",
        "Flying in before a road trip and need a Seattle store with rentals or affordable gear.",
    ),
    (
        "Is Lake City a good neighborhood?",
        "Found an apartment there and want advice about buses, groceries, and safety.",
    ),
    (
        "Recommendations for a dentist who handles anxiety",
        "Looking for a gentle Seattle dentist for someone who has avoided care.",
    ),
]

NEGATIVE_EXAMPLES = [
    (
        "Seattle City Council passes tenant protection measure",
        "The ordinance passed after hours of public comment and will take effect later this year.",
    ),
    (
        "Major delays on Link after signal issue",
        "Sound Transit says trains are single tracking between several stations this morning.",
    ),
    (
        "Found keys near Green Lake",
        "Set of keys with a blue tag found by the east path. Message with a description.",
    ),
    (
        "Capitol Hill block party street closures announced",
        "Several blocks will close from Friday morning through Sunday night for the event.",
    ),
    (
        "Photos from sunset over the Olympics tonight",
        "The clouds cleared just in time and the mountain view was unusually sharp.",
    ),
    (
        "Discussion: should Seattle expand bus-only lanes downtown?",
        "Curious how people feel after the latest SDOT proposal and public comment period.",
    ),
    (
        "Power outage in Ballard",
        "Seattle City Light map shows around 900 customers affected while crews investigate.",
    ),
    (
        "Reminder: primary ballots are due Tuesday",
        "Drop boxes close at 8 pm and mailed ballots need to be postmarked on time.",
    ),
    (
        "Local bakery workers vote to unionize",
        "Workers announced the vote after several months of organizing and negotiations.",
    ),
    (
        "Lost cat in Beacon Hill",
        "Orange tabby missing since last night near 15th Ave S. Please message if seen.",
    ),
    (
        "AMA with King County Metro planner tomorrow",
        "We will host a planner to answer questions about the service change proposal.",
    ),
    (
        "New mural completed in the CID",
        "Artists finished the wall this week after a neighborhood grant funded the project.",
    ),
    (
        "Police activity near Roosevelt Station",
        "The north entrance is taped off and buses are rerouted around the station.",
    ),
    (
        "Volunteer cleanup at Golden Gardens this Saturday",
        "Meet near the bathhouse at 10 am. Bags and gloves will be provided.",
    ),
    (
        "New protected bike lane opens on Eastlake",
        "SDOT crews finished striping and barriers this week after months of construction.",
    ),
    (
        "Seattle schools announce snow makeup day",
        "The district posted an updated calendar after last month's weather closure.",
    ),
    (
        "Smoke advisory issued for Puget Sound",
        "Public health officials recommend limiting outdoor activity through the evening.",
    ),
    (
        "Ferry schedule changes start Monday",
        "Washington State Ferries says crew availability will affect several sailings.",
    ),
    (
        "Community garden work party recap",
        "Neighbors planted native flowers and repaired the raised beds this morning.",
    ),
    (
        "Local theater announces discounted student tickets",
        "The program begins next month and applies to Wednesday performances.",
    ),
    (
        "Tree down blocking part of Dexter Ave",
        "Traffic is backing up while crews remove branches from the southbound lane.",
    ),
    (
        "King County reports increase in flu cases",
        "Health officials are encouraging vaccination and staying home when sick.",
    ),
    (
        "Public hearing scheduled for waterfront zoning proposal",
        "The planning commission will take comments at next week's meeting.",
    ),
    (
        "Sounders win at home after late goal",
        "The match ended with a stoppage-time score and a loud crowd at Lumen Field.",
    ),
    (
        "Free museum day announced for next Thursday",
        "Several museums will participate with extended hours and community programming.",
    ),
    (
        "Small business grant applications open",
        "The city says eligible neighborhood businesses can apply through the online portal.",
    ),
    (
        "Water main break closes part of Rainier Ave",
        "Crews are repairing the line and asking drivers to use alternate routes.",
    ),
    (
        "Photo: fog rolling over Elliott Bay",
        "Morning fog made the ferries look like they were disappearing into the water.",
    ),
    (
        "Ballard farmers market winter hours posted",
        "Vendors will operate from 10 am to 2 pm starting this weekend.",
    ),
    (
        "Library branch reopening after renovation",
        "The neighborhood branch will reopen with new study rooms and accessible entrances.",
    ),
    (
        "Stolen bike spotted near Fremont",
        "Posting photos and the serial number in case someone sees it around the bridge.",
    ),
    (
        "City releases update on bridge inspection",
        "Engineers found no immediate safety issue but will continue monitoring the span.",
    ),
    (
        "Election forum livestream starts at 6 pm",
        "Candidates for city council will answer questions from neighborhood groups.",
    ),
    (
        "Mariners announce new transit partnership",
        "Fans can use game tickets for discounted rides on select transit routes.",
    ),
    (
        "Air quality improves after overnight rain",
        "Monitoring stations moved back into the moderate range by early morning.",
    ),
    (
        "New salmon habitat project begins on Longfellow Creek",
        "Crews will remove invasive plants and restore streamside vegetation.",
    ),
    (
        "Reminder: no parking on parade route Sunday",
        "Temporary signs went up this week and towing starts early Sunday morning.",
    ),
    (
        "Local high school robotics team wins regional event",
        "The team will advance to the state competition after a close final round.",
    ),
    (
        "Apartment fire displaces residents in First Hill",
        "The Red Cross is assisting residents while investigators determine the cause.",
    ),
    (
        "Community meeting notes from the park redesign",
        "Neighbors discussed lighting, accessibility, playground equipment, and tree canopy.",
    ),
    (
        "Link escalator outage at University Street",
        "Transit staff say repairs are expected to continue through the weekend.",
    ),
    (
        "New public restroom opens near the waterfront",
        "The facility is staffed during daytime hours and includes drinking water access.",
    ),
    (
        "Local chef wins regional award",
        "The announcement came during Monday night's ceremony for Pacific Northwest restaurants.",
    ),
    (
        "Rain returns after unusually dry week",
        "Forecasters expect showers through Thursday with breezy conditions along the water.",
    ),
    (
        "Seattle parks department starts tree inventory",
        "Crews will survey street trees and update the public canopy database.",
    ),
    (
        "Neighborhood group files appeal of development permit",
        "The appeal cites traffic, shade, and stormwater concerns for the proposed building.",
    ),
    (
        "Found backpack on the 44 bus",
        "Black backpack with textbooks found near the rear door. Message with details.",
    ),
    (
        "Public art installation removed for repairs",
        "The sculpture will return after structural maintenance and repainting.",
    ),
    (
        "Seattle Kraken announce community rink program",
        "The team will sponsor youth skating sessions at several local rinks.",
    ),
    (
        "Road work on Aurora starts tonight",
        "Lane closures begin at 9 pm and are expected to continue for three nights.",
    ),
    (
        "Photos from the cherry blossoms at UW",
        "Campus was crowded, but the trees looked great during the afternoon sun break.",
    ),
    (
        "City budget town hall scheduled for Wednesday",
        "Residents can attend in person or watch the stream and submit comments online.",
    ),
    (
        "Buses rerouted for marathon this weekend",
        "Metro posted detours for routes crossing the race course on Sunday morning.",
    ),
    (
        "Neighborhood tool library expands hours",
        "Volunteers will open the lending desk two extra evenings each week.",
    ),
    (
        "Storm drain volunteers needed in South Park",
        "The neighborhood association is organizing a cleanup before the next heavy rain.",
    ),
    (
        "New exhibit opens at the history museum",
        "The exhibit focuses on regional labor organizing and waterfront development.",
    ),
    (
        "Earthquake drill planned for city buildings",
        "Employees and visitors may hear alarms during the scheduled exercise tomorrow.",
    ),
    (
        "Local bookstore hosting poetry reading",
        "The event starts at 7 pm and features three Seattle-area writers.",
    ),
    (
        "Update on missing dog from yesterday",
        "Good news: she was found near the park and is back home with her family.",
    ),
    (
        "Neighborhood council posts meeting agenda",
        "The agenda includes sidewalk repairs, tree canopy updates, and public comment time.",
    ),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        {"id": f"synthetic-pos-{index:03d}", "title": title, "selftext": selftext, "label": "askseattle"}
        for index, (title, selftext) in enumerate(POSITIVE_EXAMPLES, start=1)
    ]
    examples.extend(
        {
            "id": f"synthetic-neg-{index:03d}",
            "title": title,
            "selftext": selftext,
            "label": "not_askseattle",
        }
        for index, (title, selftext) in enumerate(NEGATIVE_EXAMPLES, start=1)
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    print(f"wrote {len(examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
