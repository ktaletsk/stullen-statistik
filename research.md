# stulle.app: background and reactions

Compiled 2026-11-22 from public web sources (links at the bottom). Unverified items are marked.

## Who made it

- **Gregor Weichbrodt**, credited on the site, with a link to ggor.de and a Potsdam address in the Impressum. [1][2]
  - Born 1988 in Potsdam; studied communication design at HTW Berlin. He is a conceptual / digital-literature writer. [4]
  - Works include *On the Road* (2014, Kerouac's route as Google Maps directions), *I Don't Know* (2015) and *Gruntgesets* (2016). [4]
  - Co-founded the digital-literature collective 0x0a with Hannes Bajohr; has built bots such as Zufallshorst. [6][7]
  - Recent web toys: Bahnhoroskop (06/2026) and Der Experte (07/2026). [5]
  - Socials: @gregor.bsky.social (Bluesky), @computer@chaos.social (Mastodon). [8][9]
- Music by Scott Hühnerkrisp. The site says it was inspired by the "Stullen" series of @moabitsky on Bluesky. [1][3]

## Why

- No stated motivation found. The about page presents it as just spreading a Stulle, nothing more, with no points, levels, ads or accounts. [1]

## Launch and spread

- **2026-11-21**: launched on Bluesky and Mastodon with a ~50 s trailer. He kept posting updates in a running thread:
  - paid for Cloudflare to keep it up
  - peaked at 432 concurrent users
  - deleted a swastika drawing and used Claude to flag obscene images [8][9]
- **2026-11-21**: posted to feddit c/dach (Lemmy). [10]
- **2026-11-22**: Hacker News front page, about 570 points and 198 comments; the creator commented as `greg0r`. [11]
- Picked up by aggregators (appinn and others). [12]
- No German mainstream press found. Two articles exist, but both are on AI-generated content sites.

## Reaction themes

- **Physics humour.** "Realistic cucumber behaviour" got the biggest laughs. People also enjoyed the butter spreading and found the sliding tomato unsettling.
- **Jokes.** "Better than GTA VI" and "the most German game ever" come up on Bluesky, HN and Lemmy.
- **Towers.** People share absurd stacks: dozens of pickle slices, banana and salami towers, Jenga-style fries, a Leberkäse column. This matches the height tail in the notebook.
- **Feature requests.** Mustard, pepper, lettuce, sriracha, vegan options, and a "Klappstulle" (closed sandwich) mode.
- **Problems.** Lag and overheating iPads on big stacks, breakage with script blockers, and obscene drawings that users reported.

## Tech

- Astro on Cloudflare, with Turnstile and self-hosted Plausible. A saved sandwich stores the photo plus the structured recipe (served at `/api/photos/{id}`). [2]
- From reading the JS bundle (not stated by the creator): Three.js on WebGL, photographed topping sprites, and custom bend/stack physics workers.

## Not verified

- The ddavidddev Threads post (gallery at #29,763, the perfectly round egg slice).
- Any Reddit, X or LinkedIn discussion.
- Any stated motivation.

## Sources

1. https://stulle.app/ueber
2. https://stulle.app/impressum
3. https://bsky.app/profile/moabitsky.bsky.social/post/3n27a7ffijs2d
4. https://de.wikipedia.org/wiki/Gregor_Weichbrodt
5. https://ggor.de
6. https://www.logbuch-suhrkamp.de/hannes-bajohr/0x0a-sagen/
7. https://www.dw.com/de/interview-mit-bots-gegen-die-ohnmacht/a-37064745
8. https://bsky.app/profile/gregor.bsky.social/post/3n2t2yelm7225
9. https://akko.wetdry.world/notice/117654361266845048
10. https://old.feddit.dk/post/26378577
11. https://news.ycombinator.com/item?id=50594911
12. https://www.appinn.com/stullensimulator/

## Has anyone analysed this before? (checked 2026-11-23)

I found nobody who had scraped or analysed stulle.app data. I searched GitHub repos and code, all 199 comments in the HN thread (via Algolia), the creator's Bluesky and Mastodon posts, Lemmy and the web. Reddit blocked automated access (403), so the r/de thread (https://www.reddit.com/r/de/comments/1y1ykfl/) is unchecked.

Related numbers we can compare against:

- **Creator's own counts** on Mastodon (https://chaos.social/@computer/117655442279652983):
  - 3,834 Stullen at 2026-11-21 16:46 UTC. At that moment our IDs were at ~3,970; the ~3% gap is deleted sandwiches, which matches our 404 rate.
  - About 1,000+ new per hour for most of 22 Nov; 1,750/h at the HN peak.
- **/ueber page** on 2026-11-23: 39,159 Stullen existing, newest #40,316. That's about 2.9% deleted.
- **Record towers people posted:**
  - 102 pickle slices on #28237
  - 149 salami slices (#15519 or #15714)
  - 143 radish slices on #10748
  - 42 tomato slices on #508
- **bennimmo/kids-games** on GitHub: an offline copy of the app made with the creator's permission. No analysis.

Moderation: the creator removes swastikas and Hitler drawings, usually within an hour, but leaves the penises (HN: https://news.ycombinator.com/item?id=50602792). Reported images get an LLM review.

Data caveat: blue tomatoes come from a browser bug that swaps red and blue in the photo (Firefox/Linux).

## Roll for Sandwich

A TikTok/YouTube series by Jacob Pauwels (@adventuresinaardia), a D&D writer from Grand Rapids. He rolls D&D dice against tables in six categories: bread, main, cheese, roughage, "wild magic" extras and sauce. Some rolls double a portion or force a reroll. He then builds the sandwich, tastes it and rates it out of 10. Season 2 started 2022-08-01. Sources: https://thetakeout.com/tiktok-roll-for-a-sandwich-d-and-d-style-lunch-game-1849473533 and https://www.wargamer.com/dnd/dice-roll-for-sandwich
