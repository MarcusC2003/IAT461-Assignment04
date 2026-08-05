# Client Proposal — DeckForge: New-Player Onboarding

## Entity

DeckForge is a small startup that builds a companion app for Pokémon Trading Card Game players. The app lets people track the cards they own, browse the card pool, and build decks. It's in the same space as the deck building and collection apps that already exist.

## Business Context & Strategic Options

Our engagement numbers are flat and the biggest drop off is new players in their first week. When we look at why, it's almost always the same thing: they open the app, see thousands of cards with no structure, don't know what goes with what, and leave. Management wants to add one onboarding feature next quarter to fix this and we're weighing three options:

- **Option A:** an auto-generated "archetype browser" that groups cards by how they actually play, so a new player can explore by playstyle (walls, fast attackers, engine cards, support) instead of scrolling a giant list.
- **Option B:** a price / investment tracker that flags cards likely to go up in value, to pull in the collector crowd.
- **Option C:** a "power level" warning that tells a player when a card is over- or under-costed for what it does, so they don't build a weak deck.

## The Task

**Phase 1 (EDA & Strategy Validation):** using the dataset, do an exploratory analysis to work out which of the three options this data can actually support, and say which one we should go with. The point is to not just assume, but show from the data why some options don't work. As a hint, two of them run into a wall:

- Option B needs price data and there's no price column in the dataset on purpose (card prices go stale within about a day so they're not a stable thing to build on).
- Option C needs a correct "power level" to check against and there's no such label anywhere in the data, so there's nothing to predict against.
- Option A doesn't need a price or a label, it just needs the cards' mechanics, so that's the one the data justifies.

**Phase 2 (Data Science Implementation):** assuming we go with Option A, group the cards into mechanical archetypes. There's no correct grouping already in the data and no column we're predicting — we're finding structure that's not written down, so this is a discovery task not a prediction one. The grouping can't come straight off the raw file, it needs feature engineering:

1. Parse the nested `attacks_json` and `abilities_json` into numbers (how many attacks, damage, energy cost, and something like damage per energy).
2. Mine the attack and ability text for what the card actually does (draw, search, heal, status effects, energy acceleration, etc).

Only after that do the cards fall into groups you can name.

## Dataset Provided

`pokemon_cards.csv` — 816 card records, 14 columns — built from the official public Pokémon TCG data repo (`PokemonTCG/pokemon-tcg-data`, English sets base1 to base5, gym1, gym2, neo1). I checked it loads fine in pandas. The signal you need is on purpose locked inside the `attacks_json` and `abilities_json` columns and the text, so the raw columns like hp, retreat cost and rarity on their own won't separate the archetypes — you've got to engineer the features first.

### Column dictionary

| Column | Type | What it is |
| --- | --- | --- |
| `card_id` | text | Unique card id, e.g. `base1-4` |
| `name` | text | Card name (e.g. Charizard) |
| `set_code` | categorical | Which set the card is from |
| `supertype` | categorical | Pokémon / Trainer / Energy |
| `subtypes` | categorical | Pipe separated subtypes (e.g. Stage 2) |
| `hp` | numeric | Hit points, blank for Trainer/Energy |
| `types` | categorical | Pipe separated energy type(s) |
| `evolves_from` | text | Name of the pre-evolution if there's one |
| `retreat_cost` | categorical | Pipe separated retreat energy, count it to get a number |
| `weaknesses` | categorical | Pipe separated weakness type(s) |
| `rarity` | categorical | Common / Uncommon / Rare / Rare Holo / etc |
| `attacks_json` | nested (needs parsing) | JSON string, list of attacks each with name, cost, damage, text. Parse this to get attack count, damage and energy cost (FE stage 1), and mine the text (FE stage 2). |
| `abilities_json` | nested (needs parsing) | JSON string, list of abilities each with name, type, text. Parse to flag engine/ability cards and mine the text. |
| `flavor_text` | free text | Flavour description, often blank, not really mechanical |

## Success Criteria

A good answer has two parts.

**First**, a clear EDA section that shows why Options B and C aren't doable with this data (no price column, no power-level label) so Option A is the one that makes sense — this is really about picking the right technique for what the data can support.

**Second**, for Option A, a set of card groups the Head of Product can actually use, which means:

- a small number of groups we can each give a plain name to (bulky walls, high cost heavy hitters, ability engine cards, cheap low retreat attackers, support Trainers, and so on)
- groups that come from the engineered features (parsed stats + text), not just one raw column like hp or rarity on its own
- something we can sanity check, so the data scientist says how many groups they picked and why and shows a few example cards per group

If the groups are clean enough that we could label the app's archetype browser straight from them and turn the best ones into starter collections, that's a win.

## Constraints

- Trainer and Energy cards don't have hp, type or attacks, so those are blank on purpose, not errors. Handle them on purpose (keep separate or mark them) instead of dropping them silently.
- Some cards have no attacks or no abilities (empty list) and lots have no flavour text, so the parsing shouldn't break on empty.
- Don't use rarity as the group label — it's only there for context; using it would turn this into a prediction task which isn't the point.
- The cards are from older sets so the groups reflect classic design. Prices are left out on purpose (they go stale within a day).
- Data is from a community Pokémon resource under fan content terms, it's only for this course.
