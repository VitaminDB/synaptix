# MiniMax-AI__MiniMax-H3-Image__README-examples

Source: https://github.com/MiniMax-AI/MiniMax-H3-Image
Fetched: 2026-11-28



---

## FILE: README.md (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/README.md)

<div align="center">
  <img width="100%" src="assets/minimax-h3-image.jpg" alt="MiniMax H3 image, Open Weights">
</div>

<p align="center">
  <a href="https://hailuoai.video" target="_blank"><img src="https://img.shields.io/badge/Hailuo%20AI-FF6C37?logo=minimax&logoColor=white" alt="Hailuo AI"></a>
  <a href="https://platform.minimax.io" target="_blank"><img src="https://img.shields.io/badge/API-FF6C37?logo=minimax&logoColor=white" alt="API"></a>
  <a href="https://www.minimax.io" target="_blank"><img src="https://img.shields.io/badge/MiniMax%20Website-FF6C37?logo=minimax&logoColor=white" alt="MiniMax Website"></a>
  <a href="https://github.com/MiniMax-AI/MiniMax-H3-Image" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <br>
  <a href="https://modelscope.cn/models/MiniMax/MiniMax-H3-Image" target="_blank" rel="noopener noreferrer"><img alt="ModelScope MiniMax AI" src="https://img.shields.io/badge/ModelScope-MiniMax%20AI-white?labelColor=%23EF3D5D"></a>
  <a href="https://platform.minimax.cn/docs/faq/contact-us" target="_blank"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="https://discord.com/invite/dbMxutw7tP" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-4CAF50?logo=creativecommons&logoColor=white" alt="LICENSE"></a>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.ja.md">日本語</a>
</p>

<h1 align="center">MiniMax-H3-Image</h1>

<p align="center"><b><i>One idea with open weights. A world of uses.</i></b></p>

<p align="center">
  <a href="#prompt-writing-skill">Prompt Writing Skill</a> ·
  <a href="#showcase">Showcase</a> ·
  <a href="#use-online">Use Online</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#run-locally">Run Locally</a> ·
  <a href="#prompting-tips">Prompting Tips</a> ·
  <a href="#license">License</a>
</p>

## Prompt Writing Skill

The `h3-image-prompt-writing` skill turns a short idea, a brief or a set of reference images into a
full MiniMax-H3-Image prompt, and can render it through the MiniMax API. It works in any agent that
reads a `SKILL.md`, such as Claude Code, Codex and Cursor. Install it with the
[skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add https://github.com/MiniMax-AI/MiniMax-H3-Image --skill h3-image-prompt-writing
```

Then ask your agent, for example:

```text
Use h3-image-prompt-writing to make a 3:4 poster of a night market in the rain,
titled "雨夜集市", and render it.
```

Rendering needs `MINIMAX_API_KEY`, and `MINIMAX_API_BASE=https://api.minimax.cn` on the China
platform. See [`skills/README.md`](skills/README.md) for details.

## Showcase

<!-- BEGIN showcase-grid -->
<table>
<tr><th colspan="4" align="left">Advanced Open-Source Capabilities</th></tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/multi-reference.md#sixteen-people-six-scenes"><img src="assets/showcase/tiles/multi-ref-group-a-output.jpg" alt="Sixteen people, six scenes"></a><br><sub><a href="showcase/multi-reference.md#sixteen-people-six-scenes">Sixteen people, six scenes</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/multi-reference.md#outfit-color-coordination"><img src="assets/showcase/tiles/multi-ref-outfit-color-output.jpg" alt="Outfit color coordination"></a><br><sub><a href="showcase/multi-reference.md#outfit-color-coordination">Outfit color coordination</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/multi-reference.md#living-room-from-10-references"><img src="assets/showcase/tiles/furniture-room-output.jpg" alt="Living room from 10 references"></a><br><sub><a href="showcase/multi-reference.md#living-room-from-10-references">Living room from 10 references</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#biology-figure-from-a-paper-page"><img src="assets/showcase/tiles/bio-figure-output.jpg" alt="Biology figure from a paper page"></a><br><sub><a href="showcase/instruction-following.md#biology-figure-from-a-paper-page">Biology figure from a paper page</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#red-panda-infographic"><img src="assets/showcase/tiles/red-panda-infographic.jpg" alt="Red panda infographic"></a><br><sub><a href="showcase/instruction-following.md#red-panda-infographic">Red panda infographic</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#staff-selection-puzzle"><img src="assets/showcase/tiles/staffing-logic-puzzle.jpg" alt="Staff-selection puzzle"></a><br><sub><a href="showcase/instruction-following.md#staff-selection-puzzle">Staff-selection puzzle</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#electric-motorcycle-exploded-view"><img src="assets/showcase/tiles/e-motorcycle-exploded.jpg" alt="Electric motorcycle, exploded view"></a><br><sub><a href="showcase/instruction-following.md#electric-motorcycle-exploded-view">Electric motorcycle, exploded view</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#jet-engine-exploded-view"><img src="assets/showcase/tiles/jet-engine-exploded.jpg" alt="Jet engine, exploded view"></a><br><sub><a href="showcase/instruction-following.md#jet-engine-exploded-view">Jet engine, exploded view</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#moka-pot-blueprint"><img src="assets/showcase/tiles/moka-pot-blueprint.jpg" alt="Moka pot blueprint"></a><br><sub><a href="showcase/instruction-following.md#moka-pot-blueprint">Moka pot blueprint</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#ant-inspired-hexapod-mech-line-art"><img src="assets/showcase/tiles/ant-hexapod-mech-lineart.jpg" alt="Ant-inspired hexapod mech, line art"></a><br><sub><a href="showcase/instruction-following.md#ant-inspired-hexapod-mech-line-art">Ant-inspired hexapod mech, line art</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#quadruped-mech-cad-drawing"><img src="assets/showcase/tiles/quadruped-mech-cad.jpg" alt="Quadruped mech, CAD drawing"></a><br><sub><a href="showcase/instruction-following.md#quadruped-mech-cad-drawing">Quadruped mech, CAD drawing</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#mech-beast-exploded-cad-view"><img src="assets/showcase/tiles/mech-monster-exploded-cad.jpg" alt="Mech beast, exploded CAD view"></a><br><sub><a href="showcase/instruction-following.md#mech-beast-exploded-cad-view">Mech beast, exploded CAD view</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#ceiling-rail-courier"><img src="assets/showcase/tiles/ceiling-rail-courier.jpg" alt="Ceiling rail courier"></a><br><sub><a href="showcase/instruction-following.md#ceiling-rail-courier">Ceiling rail courier</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/instruction-following.md#wheel-legged-delivery-robot"><img src="assets/showcase/tiles/wheel-leg-robot.jpg" alt="Wheel-legged delivery robot"></a><br><sub><a href="showcase/instruction-following.md#wheel-legged-delivery-robot">Wheel-legged delivery robot</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#corner-mirrors"><img src="assets/showcase/tiles/mirror-reflection.jpg" alt="Corner mirrors"></a><br><sub><a href="showcase/visual-challenges.md#corner-mirrors">Corner mirrors</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#a-hand-in-the-projection"><img src="assets/showcase/tiles/projection-hand-shadow.jpg" alt="A hand in the projection"></a><br><sub><a href="showcase/visual-challenges.md#a-hand-in-the-projection">A hand in the projection</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#game-map-to-3d-game-view"><img src="assets/showcase/tiles/game-map-to-3d.jpg" alt="Game map to 3D game view"></a><br><sub><a href="showcase/visual-challenges.md#game-map-to-3d-game-view">Game map to 3D game view</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#pelican-graduation-pyramid"><img src="assets/showcase/tiles/pelican-graduation-pyramid.jpg" alt="Pelican graduation pyramid"></a><br><sub><a href="showcase/visual-challenges.md#pelican-graduation-pyramid">Pelican graduation pyramid</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#pelican-christmas-tree"><img src="assets/showcase/tiles/pelican-christmas-tree.jpg" alt="Pelican Christmas tree"></a><br><sub><a href="showcase/visual-challenges.md#pelican-christmas-tree">Pelican Christmas tree</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/visual-challenges.md#pelican-balancing-act"><img src="assets/showcase/tiles/pelican-challenge-3.jpg" alt="Pelican balancing act"></a><br><sub><a href="showcase/visual-challenges.md#pelican-balancing-act">Pelican balancing act</a></sub></td>
</tr>
<tr><th colspan="4" align="left">Creative Applications</th></tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#frontier-rider"><img src="assets/showcase/tiles/frontier-rider.jpg" alt="Frontier rider"></a><br><sub><a href="showcase/game-worlds.md#frontier-rider">Frontier rider</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#gathering-in-the-desert"><img src="assets/showcase/tiles/desert-gathering.jpg" alt="Gathering in the desert"></a><br><sub><a href="showcase/game-worlds.md#gathering-in-the-desert">Gathering in the desert</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#cold-mineral-plain-aerial-view"><img src="assets/showcase/tiles/cold-mineral-aerial.jpg" alt="Cold mineral plain, aerial view"></a><br><sub><a href="showcase/game-worlds.md#cold-mineral-plain-aerial-view">Cold mineral plain, aerial view</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#red-mineral-plain-aerial-view"><img src="assets/showcase/tiles/desert-mineral-aerial.jpg" alt="Red mineral plain, aerial view"></a><br><sub><a href="showcase/game-worlds.md#red-mineral-plain-aerial-view">Red mineral plain, aerial view</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#forest-amphitheater"><img src="assets/showcase/tiles/forest-amphitheater.jpg" alt="Forest amphitheater"></a><br><sub><a href="showcase/game-worlds.md#forest-amphitheater">Forest amphitheater</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#palace-above-a-sea-of-clouds"><img src="assets/showcase/tiles/mountain-temple.jpg" alt="Palace above a sea of clouds"></a><br><sub><a href="showcase/game-worlds.md#palace-above-a-sea-of-clouds">Palace above a sea of clouds</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#wizard-casting-a-fan-of-fire"><img src="assets/showcase/tiles/fire-mage-vfx.jpg" alt="Wizard casting a fan of fire"></a><br><sub><a href="showcase/game-worlds.md#wizard-casting-a-fan-of-fire">Wizard casting a fan of fire</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#explosion-in-two-stages"><img src="assets/showcase/tiles/explosion-two-state.jpg" alt="Explosion in two stages"></a><br><sub><a href="showcase/game-worlds.md#explosion-in-two-stages">Explosion in two stages</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#armored-knight-character-sheet"><img src="assets/showcase/tiles/armored-knight-sheet.jpg" alt="Armored knight, character sheet"></a><br><sub><a href="showcase/game-worlds.md#armored-knight-character-sheet">Armored knight, character sheet</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#beast-king-with-a-lions-mane-close-up"><img src="assets/showcase/tiles/lion-king-face.jpg" alt="Beast king with a lion&#x27;s mane, close-up"></a><br><sub><a href="showcase/game-worlds.md#beast-king-with-a-lions-mane-close-up">Beast king with a lion&#x27;s mane, close-up</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#forest-dragon"><img src="assets/showcase/tiles/forest-dragon.jpg" alt="Forest dragon"></a><br><sub><a href="showcase/game-worlds.md#forest-dragon">Forest dragon</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#first-person-shooter-scene"><img src="assets/showcase/tiles/fps-wasteland.jpg" alt="First-person shooter scene"></a><br><sub><a href="showcase/game-worlds.md#first-person-shooter-scene">First-person shooter scene</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#untextured-render-to-film-still"><img src="assets/showcase/tiles/moon-rover-render.jpg" alt="Untextured render to film still"></a><br><sub><a href="showcase/game-worlds.md#untextured-render-to-film-still">Untextured render to film still</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#hydro-station-in-a-canyon-matte-painting"><img src="assets/showcase/tiles/hydro-station-canyon.jpg" alt="Hydro station in a canyon, matte painting"></a><br><sub><a href="showcase/game-worlds.md#hydro-station-in-a-canyon-matte-painting">Hydro station in a canyon, matte painting</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#abandoned-station"><img src="assets/showcase/tiles/abandoned-station.jpg" alt="Abandoned station"></a><br><sub><a href="showcase/game-worlds.md#abandoned-station">Abandoned station</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#industrial-ruins"><img src="assets/showcase/tiles/industrial-ruins.jpg" alt="Industrial ruins"></a><br><sub><a href="showcase/game-worlds.md#industrial-ruins">Industrial ruins</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#shipwrecks-in-a-frozen-bay"><img src="assets/showcase/tiles/frozen-shipwreck-bay.jpg" alt="Shipwrecks in a frozen bay"></a><br><sub><a href="showcase/game-worlds.md#shipwrecks-in-a-frozen-bay">Shipwrecks in a frozen bay</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#combat-pilot"><img src="assets/showcase/tiles/combat-pilot.jpg" alt="Combat pilot"></a><br><sub><a href="showcase/game-worlds.md#combat-pilot">Combat pilot</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#polar-outfit-costume-board"><img src="assets/showcase/tiles/polar-costume-board.jpg" alt="Polar outfit, costume board"></a><br><sub><a href="showcase/game-worlds.md#polar-outfit-costume-board">Polar outfit, costume board</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#helmet-on-a-red-plain-key-art"><img src="assets/showcase/tiles/helmet-key-art.jpg" alt="Helmet on a red plain, key art"></a><br><sub><a href="showcase/game-worlds.md#helmet-on-a-red-plain-key-art">Helmet on a red plain, key art</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#sci-fi-fighter"><img src="assets/showcase/tiles/scifi-fighter.jpg" alt="Sci-fi fighter"></a><br><sub><a href="showcase/game-worlds.md#sci-fi-fighter">Sci-fi fighter</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/game-worlds.md#modified-muscle-car"><img src="assets/showcase/tiles/modded-muscle-car.jpg" alt="Modified muscle car"></a><br><sub><a href="showcase/game-worlds.md#modified-muscle-car">Modified muscle car</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#farm-in-the-fog-documentary-film-photo"><img src="assets/showcase/tiles/farm-documentary-film.jpg" alt="Farm in the fog, documentary film photo"></a><br><sub><a href="showcase/real-world.md#farm-in-the-fog-documentary-film-photo">Farm in the fog, documentary film photo</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#picnic-on-film"><img src="assets/showcase/tiles/picnic-film.jpg" alt="Picnic on film"></a><br><sub><a href="showcase/real-world.md#picnic-on-film">Picnic on film</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#close-up-portrait-on-film"><img src="assets/showcase/tiles/film-portrait-closeup.jpg" alt="Close-up portrait on film"></a><br><sub><a href="showcase/real-world.md#close-up-portrait-on-film">Close-up portrait on film</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#snapshot-from-an-old-phone"><img src="assets/showcase/tiles/iphone7-snapshot.jpg" alt="Snapshot from an old phone"></a><br><sub><a href="showcase/real-world.md#snapshot-from-an-old-phone">Snapshot from an old phone</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#flash-portrait-on-film"><img src="assets/showcase/tiles/flash-film-portrait.jpg" alt="Flash portrait on film"></a><br><sub><a href="showcase/real-world.md#flash-portrait-on-film">Flash portrait on film</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#grandmas-birthday-home-video"><img src="assets/showcase/tiles/grandma-birthday-dv.jpg" alt="Grandma&#x27;s birthday, home video"></a><br><sub><a href="showcase/real-world.md#grandmas-birthday-home-video">Grandma&#x27;s birthday, home video</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#rooftop-at-dusk"><img src="assets/showcase/tiles/rooftop-bedsheets.jpg" alt="Rooftop at dusk"></a><br><sub><a href="showcase/real-world.md#rooftop-at-dusk">Rooftop at dusk</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#a-pause-before-tea"><img src="assets/showcase/tiles/tea-portrait.jpg" alt="A pause before tea"></a><br><sub><a href="showcase/real-world.md#a-pause-before-tea">A pause before tea</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/real-world.md#flamingos-at-dawn"><img src="assets/showcase/tiles/flamingo-salt-lake.jpg" alt="Flamingos at dawn"></a><br><sub><a href="showcase/real-world.md#flamingos-at-dawn">Flamingos at dawn</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#a-cloud-in-a-barber-chair"><img src="assets/showcase/tiles/surreal-cloud-barber-chair.jpg" alt="A cloud in a barber chair"></a><br><sub><a href="showcase/surreal-worlds.md#a-cloud-in-a-barber-chair">A cloud in a barber chair</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#slicing-the-moon"><img src="assets/showcase/tiles/surreal-long-table.jpg" alt="Slicing the moon"></a><br><sub><a href="showcase/surreal-worlds.md#slicing-the-moon">Slicing the moon</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#a-pear-as-a-dressing-room"><img src="assets/showcase/tiles/surreal-pear-dressing-room.jpg" alt="A pear as a dressing room"></a><br><sub><a href="showcase/surreal-worlds.md#a-pear-as-a-dressing-room">A pear as a dressing room</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#cut-paper-collage"><img src="assets/showcase/tiles/blue-collage.jpg" alt="Cut-paper collage"></a><br><sub><a href="showcase/surreal-worlds.md#cut-paper-collage">Cut-paper collage</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#sticker-residue-type-poster"><img src="assets/showcase/tiles/sticker-type-poster.jpg" alt="Sticker-residue type poster"></a><br><sub><a href="showcase/surreal-worlds.md#sticker-residue-type-poster">Sticker-residue type poster</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/surreal-worlds.md#pixel-cat-poster"><img src="assets/showcase/tiles/pixel-white-cat.jpg" alt="Pixel cat poster"></a><br><sub><a href="showcase/surreal-worlds.md#pixel-cat-poster">Pixel cat poster</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#mandarin-ducks-old-master-oil-painting"><img src="assets/showcase/tiles/baroque-mandarin-ducks.jpg" alt="Mandarin ducks, Old Master oil painting"></a><br><sub><a href="showcase/artistic-worlds.md#mandarin-ducks-old-master-oil-painting">Mandarin ducks, Old Master oil painting</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#fishermen-and-a-pelican-caravaggio-style"><img src="assets/showcase/tiles/caravaggio-pelican.jpg" alt="Fishermen and a pelican, Caravaggio style"></a><br><sub><a href="showcase/artistic-worlds.md#fishermen-and-a-pelican-caravaggio-style">Fishermen and a pelican, Caravaggio style</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#table-tennis-caravaggio-style"><img src="assets/showcase/tiles/caravaggio-ping-pong.jpg" alt="Table tennis, Caravaggio style"></a><br><sub><a href="showcase/artistic-worlds.md#table-tennis-caravaggio-style">Table tennis, Caravaggio style</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#mandarin-ducks-in-a-suzhou-garden-impressionist-oil-painting"><img src="assets/showcase/tiles/impressionist-suzhou-garden.jpg" alt="Mandarin ducks in a Suzhou garden, Impressionist oil painting"></a><br><sub><a href="showcase/artistic-worlds.md#mandarin-ducks-in-a-suzhou-garden-impressionist-oil-painting">Mandarin ducks in a Suzhou garden, Impressionist oil painting</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#the-sun-illuminated-manuscript"><img src="assets/showcase/tiles/illuminated-manuscript-a.jpg" alt="The Sun, illuminated manuscript"></a><br><sub><a href="showcase/artistic-worlds.md#the-sun-illuminated-manuscript">The Sun, illuminated manuscript</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#the-jade-rabbit-illuminated-manuscript"><img src="assets/showcase/tiles/illuminated-manuscript-b.jpg" alt="The Jade Rabbit, illuminated manuscript"></a><br><sub><a href="showcase/artistic-worlds.md#the-jade-rabbit-illuminated-manuscript">The Jade Rabbit, illuminated manuscript</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#space-station-repair-dunhuang-mural"><img src="assets/showcase/tiles/dunhuang-space-station.jpg" alt="Space station repair, Dunhuang mural"></a><br><sub><a href="showcase/artistic-worlds.md#space-station-repair-dunhuang-mural">Space station repair, Dunhuang mural</a></sub></td>
</tr>
<tr>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#pelican-shadow-puppet"><img src="assets/showcase/tiles/shadow-puppet-pelican.jpg" alt="Pelican shadow puppet"></a><br><sub><a href="showcase/artistic-worlds.md#pelican-shadow-puppet">Pelican shadow puppet</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#circuit-board-textile-gongbi"><img src="assets/showcase/tiles/gongbi-circuit-textile.jpg" alt="Circuit-board textile, gongbi"></a><br><sub><a href="showcase/artistic-worlds.md#circuit-board-textile-gongbi">Circuit-board textile, gongbi</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#pelican-at-a-mirror-gongbi-painting"><img src="assets/showcase/tiles/gongbi-pelican-mirror.jpg" alt="Pelican at a mirror, gongbi painting"></a><br><sub><a href="showcase/artistic-worlds.md#pelican-at-a-mirror-gongbi-painting">Pelican at a mirror, gongbi painting</a></sub></td>
<td align="center" valign="top" width="25%"><a href="showcase/artistic-worlds.md#golden-flames-eastern-fantasy-illustration"><img src="assets/showcase/tiles/eastern-fantasy-lineart.jpg" alt="Golden flames, Eastern fantasy illustration"></a><br><sub><a href="showcase/artistic-worlds.md#golden-flames-eastern-fantasy-illustration">Golden flames, Eastern fantasy illustration</a></sub></td>
</tr>
</table>

Select an image to see it larger with its prompt, or browse [by section](showcase/README.md).
<!-- END showcase-grid -->

## Use Online

| | Global | China |
| --- | --- | --- |
| Web app | [hailuoai.video](https://hailuoai.video/) | [hailuoai.com](https://hailuoai.com/) |
| Desktop app | [agent.minimax.io](https://agent.minimax.io/) | [agent.minimax.cn](https://agent.minimax.cn) |
| API platform | [platform.minimax.io](https://platform.minimax.io/docs/guides/image-generation) | [platform.minimax.cn](https://platform.minimax.cn/docs/guides/image-generation) |
| API reference | [Text-to-image](https://platform.minimax.io/docs/api-reference/image-generation-h3-generations) · [Image editing](https://platform.minimax.io/docs/api-reference/image-generation-h3-edits) · [Prompt expansion](https://platform.minimax.io/docs/api-reference/image-generation-h3-context-ir) | [Text-to-image](https://platform.minimax.cn/docs/api-reference/image-generation-h3-generations) · [Image editing](https://platform.minimax.cn/docs/api-reference/image-generation-h3-edits) · [Prompt expansion](https://platform.minimax.cn/docs/api-reference/image-generation-h3-context-ir) |

## Overview

MiniMax-H3-Image is a state-of-the-art open-source model for image generation and editing from the
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) family, with text-to-image and
multi-image editing in one checkpoint.

- **16 MP at a practical cost.** It generates images of about 16 MP at every aspect ratio directly,
  without an upscaler, in 6 sampling steps.
- **Rich detail, real-world realism.** Its images go beyond the familiar "AI look".
- **Strongest instruction following among open-source models.** It takes up to 16 reference images.

### Specifications

These are the specifications of the MiniMax API. A local server takes the fields in
[Request Parameters](#request-parameters).

| Parameter | Specification |
| --- | --- |
| Model ID | `MiniMax-H3-Image` |
| Tasks | Text-to-image, image editing |
| Image size | `width` + `height`, or `area` + `aspect_ratio`. Default `auto`: 4K area, automatic aspect ratio |
| Output area | `1K`, `2K` or **`4K` (best results 👍)**; a custom `width` × `height` covers 655,360 to 20,000,000 pixels |
| Aspect ratio | 1:3 to 3:1 |
| Prompt | Up to 32,000 characters; content beyond 4,096 tokens is truncated |
| Reference images | Up to 16, each at most 64 MiB, with sides of 128 to 8192 pixels and an aspect ratio from 1:4 to 4:1 |
| Reference formats | JPG, PNG, WebP, HEIC or HEIF, sent as an HTTP(S) URL, Base64 or MiniMax file ID |
| Images per request | 1 to 10 |
| Prompt expansion | On by default for text-to-image and can be turned off; always on for image editing; the expanded prompt can be returned |
| Output | PNG (default), JPEG or WebP, with `output_compression` from 0 to 100; returned as a URL (default) or Base64 |
| Not yet supported | Mask-based inpainting, transparent backgrounds, user-specified seeds, preview images |

### Model Zoo

| Model | Role | Precision | Size | Hugging Face |
| --- | --- | --- | --- | --- |
| **MiniMax-H3-Image** | Text-to-image and image editing | BF16 (DiT) | 68.65 GB | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) |
| **MiniMax-H3-Image-PE** (optional) | Prompt expander for text-to-image | NVFP4 | 17.9 GB | [MiniMaxAI/MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) |

### Model Architecture

MiniMax-H3-Image follows the H3-Base design with the audio branch removed. The text prompt and the
reference images are encoded by the H3 text encoder; the reference images are also encoded by the
image VAE. All tokens form one packed sequence for a single-stream transformer, which predicts the
latent of the target image. The VAE decodes that latent into the image.

- **Text encoder**: the full Qwen3-VL text encoder of MiniMax-H3, unchanged. The DiT reads the
  hidden states of its 50th layer.
- **Image VAE**: a CNN encoder and a 2D ViT decoder (24 layers), with 16x spatial downsampling and
  24 latent channels. The DiT patchifies the latent with a 2x2 patch, so each transformer token
  covers 32x32 pixels.
- **Diffusion transformer**: 33.1B parameters, a dense single-stream transformer with 50 layers
  (hidden size 5376, 56 attention heads, FFN size 14336) and 3D multimodal RoPE, the same as the
  H3-Omni-Transformer. The weights are BF16; the embedding, timestep and output layers are FP32.

## Run Locally

### Download

| Component | Content | Size |
| --- | --- | --- |
| `transformer/` | DiT weights. | 66.28 GB |
| `pdd_fused_heads.safetensors`, `pdd_config.json` | The per-step output heads that replace the transformer's output layer, and the sampling schedule. | 12 MB |
| `vae/` | Image VAE, FP32. | 2.34 GB |
| `tokenizer/`, `processor/` | Qwen3-VL tokenizer and processor, identical to MiniMax-H3. | 23 MB |
| Text encoder | Not part of the original checkpoint. `model_index.json` loads it from [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3), folder `text_encoder/`. If you already have MiniMax-H3, it is not downloaded again. | 66.7 GB |

The Hugging Face repository holds this original checkpoint alongside the diffusers format, about
204 GB in total. SGLang and vLLM-Omni need only the original checkpoint and the MiniMax-H3 text
encoder, about 135 GB:

```bash
hf download MiniMaxAI/MiniMax-H3-Image model_index.json "pdd_*" "processor/*" "tokenizer/*" \
  "transformer/*" "vae/*"
hf download MiniMaxAI/MiniMax-H3 --include "text_encoder/*"
# optional prompt expander
hf download MiniMaxAI/MiniMax-H3-Image-PE
```

diffusers fetches its own files with `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image")`.

### Deployment

We recommend the following inference frameworks to serve the model:

- [SGLang](https://docs.sglang.io/) - see [cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - see [vllm-omni recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)
- [diffusers](https://github.com/huggingface/diffusers) - see [diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) - see [Comfy tutorial](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image); use [T2I template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json) / [Edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json)

#### SGLang

Install SGLang from the main branch with its diffusion dependencies
([installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation)):

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
python3 -m pip install -e 'python[diffusion]'
```

Serve on one GPU (SGLang offloads model components when less than 130 GiB is free), or split each
request over several GPUs with sequence parallelism (Ulysses) for lower latency:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --host 0.0.0.0 --port 30000 --warmup-mode off
# 4 GPUs
sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --host 0.0.0.0 --port 30000 --warmup-mode off \
  --num-gpus 4 --ulysses-degree 4
```

Add `--text-encoder-path <path to text_encoder>` to use a local copy of the MiniMax-H3 text encoder.
The [SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image) has
recipes for other GPUs, from B300 down to 12 GB cards.

#### vLLM-Omni

Install vLLM 0.33.0, then vLLM-Omni from source:

```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm==0.33.0 --torch-backend=cu130
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni && uv pip install -e .
```

Serve on 4 GPUs with sequence parallelism (Ulysses), or on one GPU: a B200 or B300 holds the whole
model in about 135 GiB, and one H200 offloads the text encoder:

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni --host 0.0.0.0 --port 8000 \
  --num-gpus 4 --usp 4 --text-encoder-tp-size 4
# one B200 or B300
vllm serve MiniMaxAI/MiniMax-H3-Image --omni --host 0.0.0.0 --port 8000 --num-gpus 1
# one H200: offload the text encoder
vllm serve MiniMaxAI/MiniMax-H3-Image --omni --host 0.0.0.0 --port 8000 --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'
```

The [vLLM-Omni recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image) covers AMD Instinct and
other GPUs.

### Request Parameters

SGLang takes text-to-image requests as JSON at `/v1/images/generations` and image editing requests
as multipart form data at `/v1/images/edits`; both return the PNG in `data[0].b64_json`. vLLM-Omni
takes slightly different fields; see its [recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image).

| Field | Text-to-image | Image editing | Notes |
| --- | --- | --- | --- |
| `prompt` | Required | Required | Used as given, truncated to 4096 tokens |
| `size` | Required | Required | `WIDTHxHEIGHT`. The returned image has exactly this size; see [Sizes](#sizes) |
| `image` | — | One or more files | In upload order they are `<Picture 1>`, `<Picture 2>`, ... in the prompt |
| `seed` | Recommended | Recommended | Integer |
| `generator_device` | `cpu` recommended | `cpu` recommended | Same seed, same initial noise on every machine |
| `response_format` | `b64_json` | `b64_json` | |
| `min_generation_area` | Optional | Optional | Minimum generated area in pixels, 1024-16777216, default 4194304 (2048x2048). A smaller `size` is generated at the same aspect ratio on a larger canvas and resized back with LANCZOS; pass 1024 to turn this off. All sizes in the table below are above the default |
| `num_inference_steps` | Omit or 6 | Omit or 6 | Other values return HTTP 400 |
| `guidance_scale` | Omit or 1 | Omit or 1 | Other values return HTTP 400 |
| `negative_prompt` | Not supported | Not supported | Rejected |

### Sizes

The model works at about 4 MP to 20 MP and generally gives the best results at about 16 MP. Common
16 MP sizes:

| Aspect ratio | Size (width x height) |
| --- | --- |
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

For another aspect ratio, keep about 16 MP with both sides a multiple of 32. `size_for()` in
[`examples/context_ir.py`](examples/context_ir.py) computes it:

```bash
cd MiniMax-H3-Image/examples
python3 -c "from context_ir import size_for; print(size_for('5:4'))"   # 4576x3648
```

### Full Workflow

The MiniMax API expands a short prompt in any language into a detailed English description of the
whole image with **H3-Image-Context-IR** before generating. A local server uses the prompt as
given, so expand it first, as in the cases below. The scripts in [`examples/`](examples) reproduce
the API results with the SGLang server above, using only the Python standard library:

```bash
git clone https://github.com/MiniMax-AI/MiniMax-H3-Image.git
cd MiniMax-H3-Image/examples
export MINIMAX_API_BASE="https://api.minimax.io"   # China: https://api.minimax.cn
export MINIMAX_API_KEY="<api-key>"
```

`t2i.py` and `edit.py` send requests to `http://127.0.0.1:30000`; pass `--url` to use another
address.

#### case-T2I

[`instruction.t2i.txt`](examples/instruction.t2i.txt).

<table>
<thead>
<tr><th colspan="2">Stage</th><th>Request</th><th>Result</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Prompt Enhancement</td><td>H3-Image-Context-IR (MiniMax API)</td><td><code>python3 context_ir.py t2i</code> (<a href="examples/context_ir.py">script</a>)</td><td><code>revised_prompt.txt</code>, <code>revised_size.txt</code>; our run: <a href="examples/revised_prompt.t2i.txt"><code>revised_prompt.t2i.txt</code></a>, <a href="examples/revised_size.t2i.txt"><code>revised_size.t2i.txt</code></a> (4736x3552)</td></tr>
<tr><td>local alternative: <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE">MiniMax-H3-Image-PE</a></td><td><code>scripts/expand_prompt.py</code> in the PE repository</td><td>Save <code>expanded_caption</code> as <code>revised_prompt.txt</code> and the size for <code>aspect_ratio</code> as <code>revised_size.txt</code></td></tr>
<tr><td>agent alternative: <a href="#prompt-writing-skill">h3-image-prompt-writing</a></td><td>Ask your agent to write a prompt and an aspect ratio for <code>instruction.t2i.txt</code> with the skill</td><td>Save the prompt as <code>revised_prompt.txt</code> and the <a href="#sizes">size</a> for that aspect ratio as <code>revised_size.txt</code></td></tr>
<tr><td colspan="2">MiniMax-H3-Image generation: local SGLang</td><td><code>python3 t2i.py --prompt revised_prompt.txt --size $(cat revised_size.txt)</code> (<a href="examples/t2i.py">script</a>)</td><td><code>t2i.png</code></td></tr>
<tr><td colspan="2">E2E API Reference: text-to-image by directly calling the MiniMax API</td><td><code>python3 api_generate.py t2i</code> (<a href="examples/api_generate.py">script</a>)</td><td><code>api_t2i.png</code></td></tr>
</tbody>
</table>

<p align="center">
<img src="assets/workflow_t2i.jpg" alt="Left: t2i.png from local SGLang with the enhanced prompt. Right: api_t2i.png from the MiniMax API. Both show an overhead farm-to-table feast with bread, vegetables and cheeses on a weathered wooden table" width="100%"/>
</p>
<p align="center"><em>t2i.png from local SGLang (left) and api_t2i.png from the MiniMax API (right), 4736x3552 (4:3 picked by H3-Image-Context-IR)</em></p>

#### case-Edit

[`instruction.edit.txt`](examples/instruction.edit.txt) with the reference image
[`reference1.png`](examples/reference1.png) as `<Picture 1>`.

| Stage | Request | Result |
| --- | --- | --- |
| Prompt Enhancement: H3-Image-Context-IR (MiniMax API) | `python3 context_ir.py edit` ([script](examples/context_ir.py)) | `revised_prompt.txt`, `revised_size.txt`; our run: [`revised_prompt.edit.txt`](examples/revised_prompt.edit.txt), [`revised_size.edit.txt`](examples/revised_size.edit.txt) (4096x4096) |
| Prompt Enhancement, agent alternative: [h3-image-prompt-writing](#prompt-writing-skill) | Ask your agent to rewrite `instruction.edit.txt` for `reference1.png` with the skill | Save the prompt as `revised_prompt.txt` and `4096x4096` as `revised_size.txt` |
| MiniMax-H3-Image generation: local SGLang | `python3 edit.py --prompt revised_prompt.txt --size $(cat revised_size.txt)` ([script](examples/edit.py)) | `edit.png` |
| E2E API Reference: image editing by directly calling the MiniMax API | `python3 api_generate.py edit` ([script](examples/api_generate.py)) | `api_edit.png` |

<p align="center">
<img src="assets/workflow_edit.jpg" alt="Left: reference1.png, the reference image of two podcast hosts with the man wearing headphones. Middle: edit.png from local SGLang with the enhanced prompt. Right: api_edit.png from the MiniMax API. In both results the headphones are gone and the hosts bump fists above the MONEY TALKS sign" width="100%"/>
</p>
<p align="center"><em>Reference image reference1.png (left), edit.png from local SGLang (middle) and api_edit.png from the MiniMax API (right), 4096x4096 (1:1 picked by H3-Image-Context-IR)</em></p>

## Prompting Tips

These tips from the official user guide apply to the MiniMax API and to a local server alike; the
[Prompt Writing Skill](#prompt-writing-skill) follows them for you.

- **Task before style, layout before details.** Say what the image is for and what is in it, then
  the layout, then the style through visible color, material and light instead of words like
  "premium".
- **Quote every word to show**, with its position and line breaks. Write "no text, letters or
  numbers in the image" when you want none.
- **Give exact data** for charts and educational images, with units and time ranges, and mark
  example data as such.
- **Give each reference image a role**, and say what changes and what stays: "Change the cup on the
  left to blue, keeping its shape, material, position and size."
- **Write grids as rows and columns**, such as "2 rows and 3 columns", not "2×3".

## License

- MiniMax-H3-Image is released under the [MiniMax-H3-Image Community License Agreement](LICENSE).
- The T2I PE Model ([MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE)) is part of MiniMax-H3-Image and is accordingly subject to [MiniMax-H3-Image Community License Agreement](LICENSE).
- The separate written authorization described in Section IV of the license is requested through [Application form](https://platform.minimax.io/console/license)

## Contact Us

Contact us at [model@minimax.io](mailto:model@minimax.io).


---

## FILE: examples/instruction.edit.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/instruction.edit.txt)

```
Take the over-ear headphones off the man on the left and fill in his hair where the band flattened it. Have both hosts lean toward each other and fist-bump directly above the "MONEY TALKS" desk sign, knuckles touching, elbows lifted off the desk, so the bump sits dead centre between the two microphones.

```


---

## FILE: examples/instruction.t2i.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/instruction.t2i.txt)

```
A rustic, overhead shot of a farm-to-table feast, showcasing a variety of colorful, seasonal vegetables, artisanal cheeses, and freshly baked bread on a weathered wooden table.

```


---

## FILE: examples/prompt.edit.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/prompt.edit.txt)

```
The dark, warmly lit podcast studio photograph of two hosts behind a wooden desk in <Picture 1> is <Subject 1>. <Subject 1> is mostly preserved: the large "EP. 47" title in the top right corner, the dark textured wall, the wooden shelves with plants, a brass globe, a lit candle, books and the "DISCIPLINE > MOTIVATION" chalkboard, the framed "COMPOUND CONSISTENCY CONFIDENCE" poster, the glowing floor lamp on the left and the table lamp on the right, the two black microphones on their stands, the black "INVEST IN YOURSELF" and "PLAN BUILD WEALTH" mugs, the carved wooden "MONEY TALKS" desk sign, the dark wooden desk, both hosts' faces, smiles and clothing, the man's wristwatch and the woman's hoop earrings all stay exactly the same. Only the headphones and the hosts' poses change. The over-ear headphones and their cable are removed from the man on the left, and his short, tousled dark brown curly hair is filled in naturally over the top and sides of his head where the headband had flattened it, leaving no dent or mark. The two hosts lean toward each other and bump fists directly above the "MONEY TALKS" sign: one fist from each host meets with the knuckles touching, both elbows lifted off the desk, so the fist bump sits exactly at the horizontal centre of the frame, midway between the two microphones. The target image is a square, cinematic photograph of the podcast set: the man in the black crew-neck sweater on the left and the woman with long wavy brown hair in the cream knit sweater on the right lean in and smile at each other, their forearms raised so the touching fists hover just above the top edge of the "MONEY TALKS" sign between the two microphones, while the warm lamp light, the dark shelves behind them and the "EP. 47" title in the top right keep the original moody lighting and colors.
```


---

## FILE: examples/request.t2i.json (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/request.t2i.json)

```
{
  "prompt": "The image is a photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across a heavily weathered, dark brown wooden table with deep visible grain and scratches. The composition is a top-down view looking directly at the table surface. In the center sits a large woven jute placemat supporting a freshly baked rustic sourdough loaf with a deeply scored, golden-brown crust and a dusting of flour. To the upper left, a hammered copper bowl overflows with a colorful mix of fresh seasonal vegetables, including purple kale, bright orange carrots, red beets, and vibrant green salad leaves. In the upper right, a dark wooden cheese board holds neatly arranged artisanal cheeses: a wedge of bloomy rind brie with a creamy yellow interior, a wedge of aged cheddar with a golden rind, and a log of goat cheese sprinkled with cracked peppercorns and rosemary sprigs. The lower left contains a cluster of heirloom tomatoes, small cherry tomatoes on the vine, scattered toasted walnuts, and a small ceramic bowl of green olives. In the lower right, a clear glass wine glass filled with dark red wine sits beside a vintage brass candle holder with a lit white taper candle. A smaller wooden cutting board holding sliced apples and a wedge of brie overlaps the center of the table. The warm natural sunlight originates from the left, creating a realistic interplay of dappled shadows, crisp highlights on the polished glass and glossy tomatoes, and rich drop shadows that emphasize the rough wood grain and textured vegetables.",
  "size": "4736x3552",
  "seed": 42,
  "generator_device": "cpu",
  "response_format": "b64_json",
  "output_format": "png"
}
```


---

## FILE: examples/revised_prompt.edit.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/revised_prompt.edit.txt)

```
The man on the left in a dark sweater and wristwatch in <Picture 1> is <Subject 1>. The woman on the right in a beige sweater and hoop earrings in <Picture 1> is <Subject 2>. The pair of black microphones on stands in front of the hosts in <Picture 1> is <Subject 3>. The wooden desk sign with raised lettering reading 'MONEY TALKS' in <Picture 1> is <Subject 4>. The black mug on the left with text reading 'INVEST', 'IN', and 'YOURSELF' on three lines in <Picture 1> is <Subject 5>. The black mug on the right with text reading 'PLAN', 'BUILD', and 'WEALTH' on three lines in <Picture 1> is <Subject 6>. The upper-right overlay text reading 'EP. 47' in <Picture 1> is <Subject 7>. The framed wall poster on the right reading 'COMPOUND', 'CONSISTENCY', and 'CONFIDENCE' on three lines in <Picture 1> is <Subject 8>. The small framed shelf sign in the center background reading 'DISCIPLINE', '>', and 'MOTIVATION' on three lines in <Picture 1> is <Subject 9>. The podcast studio environment, including the wooden desk, shelves, lamps, books, and decor in <Picture 1>, is <Subject 10>. The target image shows <Subject 1> without the over-ear headphones from <Picture 1> and with fuller hair, as <Subject 1> and <Subject 2> lean inward to fist-bump directly above <Subject 4> between <Subject 3>. <Subject 1> is partially preserved; his identity, facial features, expression, dark sweater, and wristwatch remain, while his headphones are removed, his crown hair is filled in, and his posture changes from resting his forearms on the desk to leaning toward <Subject 2> with lifted elbows for a central fist-bump. <Subject 2> is partially preserved; her identity, facial features, expression, hair, earrings, and beige sweater remain, while her posture changes from crossed arms on the desk to leaning toward <Subject 1> with lifted elbows for a central fist-bump. <Subject 3> is fully preserved. <Subject 4> is fully preserved. <Subject 5> is fully preserved. <Subject 6> is fully preserved. <Subject 7> is fully preserved. <Subject 8> is fully preserved. <Subject 9> is fully preserved. <Subject 10> is fully preserved. On <Subject 1>, the headband indentation across the top of his head is gone, replaced by natural, voluminous wavy brown curls that match the texture and warm highlights of his existing hair, leaving his ears and temples completely uncovered where the earcups and cords used to be. Both <Subject 1> and <Subject 2> angle their torsos slightly inward toward the center of the frame. Their elbows are raised clear off the desk surface, bringing their inner forearms up into the space behind <Subject 3> and directly above <Subject 4>. Their hands are clenched into fists with knuckles touching dead center between the two microphones of <Subject 3>, casting soft contact shadows onto each other's knuckles and subtle warm light reflections from the studio lamps across their sleeves and raised arms.
```


---

## FILE: examples/revised_prompt.t2i.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/revised_prompt.t2i.txt)

```
The image is an overhead, top-down photograph of a rustic farm-to-table feast spread across a weathered, dark brown wooden table with visible grain, knots, and a worn finish. In the center, a large woven basket holds a freshly baked rustic loaf of bread with a cracked golden-brown crust and scattered flour, alongside a halved loaf. To the left, a large dark ceramic plate presents an assortment of colorful seasonal vegetables, including bright orange carrots, deep red heirloom tomatoes, purple cabbage, and a mound of vibrant green leafy kale and arugula. On the right side, a wooden board displays artisanal cheeses: a wedge of sharp aged cheddar, a round of creamy brie with a rind, a soft blue cheese, and a few scattered almonds and walnuts. Below the bread in the lower center, a shallow bowl contains roasted root vegetables, including golden potatoes and beets, sprinkled with fresh rosemary sprigs. At the upper right, a glass jar of golden olive oil stands beside a smaller jar of honey with a wooden dipper. In the lower left, a rustic cloth napkin holds a silver knife and fork. At the upper left, a bunch of purple and red grapes rests near a small bowl of black olives. Scattered herbs, including fresh thyme and sprigs of oregano, decorate the table. The lighting is natural and warm, coming from the upper left and casting soft, long shadows to the lower right, enhancing the texture of the bread, the moisture of the vegetables, and the overall cozy, bountiful atmosphere of the meal.
```


---

## FILE: examples/revised_size.edit.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/revised_size.edit.txt)

```
4096x4096
```


---

## FILE: examples/revised_size.t2i.txt (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/revised_size.t2i.txt)

```
4736x3552
```


---

## FILE: examples/context_ir.py (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/context_ir.py)

```
#!/usr/bin/env python3
"""Enhance a prompt with the MiniMax H3-Image-Context-IR API.

    export MINIMAX_API_BASE=https://api.minimax.cn   # CN; Global: https://api.minimax.io
    export MINIMAX_API_KEY=<API key>
    python3 context_ir.py t2i    # instruction.t2i.txt
    python3 context_ir.py edit   # instruction.edit.txt with reference1.png

Sends the instruction (and, for edit, the reference images) to POST /v1/h3_image_context_ir
and writes the enhanced prompt to revised_prompt.txt. Unless --size is given, the API also
picks the aspect ratio; the matching size of about 16 megapixels is written to
revised_size.txt. Pass the files to t2i.py / edit.py with --prompt and --size. Uses only the
Python standard library.
"""

import argparse
import base64
import json
import math
import mimetypes
import os
import urllib.request
from pathlib import Path

# absolute(), not resolve(): in the Hugging Face cache this file is a symlink into blobs/.
HERE = Path(__file__).absolute().parent
AREA = 4096 * 4096


def data_uri(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0]
    if content_type is None:
        raise ValueError(f"unknown image type: {path}")
    return f"data:{content_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def size_for(aspect_ratio: str) -> str:
    """About 16 megapixels at the ratio, both sides multiples of 32 (matches the README Sizes table)."""
    x, y = (int(v) for v in aspect_ratio.split(":"))
    width = round(math.sqrt(AREA * x / y) / 32) * 32
    height = round(math.sqrt(AREA * y / x) / 32) * 32
    return f"{width}x{height}"


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{os.environ['MINIMAX_API_BASE'].rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def request_body(task: str, instruction: Path, images: list, size) -> dict:
    body = {"model": "MiniMax-H3-Image", "prompt": instruction.read_text().strip()}
    if task == "edit":
        body["images"] = [{"image_url": data_uri(path)} for path in images or [HERE / "reference1.png"]]
    if size:
        body["size"] = size
    return body


def add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task", choices=["t2i", "edit"])
    parser.add_argument("--instruction", type=Path, help="default: instruction.<task>.txt")
    parser.add_argument("--image", action="append", type=Path,
                        help="edit only: reference image; repeat for <Picture 2>, ... (default: reference1.png)")
    parser.add_argument("--size", help="WIDTHxHEIGHT of the image you will generate (default: chosen by the API)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_arguments(parser)
    parser.add_argument("--out", default="revised_prompt.txt")
    parser.add_argument("--size-out", default="revised_size.txt", help="written when the API picks the aspect ratio")
    args = parser.parse_args()

    instruction = args.instruction or HERE / f"instruction.{args.task}.txt"
    result = post("/v1/h3_image_context_ir", request_body(args.task, instruction, args.image, args.size))
    Path(args.out).write_text(result["revised_prompt"])
    summary = {"request_id": result.get("request_id"), "usage": result.get("usage")}
    if "aspect_ratio" in result:
        summary["aspect_ratio"] = result["aspect_ratio"]
        summary["size"] = size_for(result["aspect_ratio"])
        Path(args.size_out).write_text(summary["size"])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

```


---

## FILE: examples/t2i.py (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/t2i.py)

```
#!/usr/bin/env python3
"""Send request.t2i.json to a MiniMax-H3-Image server and write the PNG.

    python3 t2i.py [--url http://127.0.0.1:30000] [--prompt FILE] [--size WxH] [--out t2i.png]

--prompt and --size replace the prompt and the size in the request.

Uses only the Python standard library.
"""

import argparse
import base64
import json
import urllib.request
from pathlib import Path

# absolute(), not resolve(): in the Hugging Face cache this file is a symlink into blobs/.
HERE = Path(__file__).absolute().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30000", help="server base URL")
    parser.add_argument("--request", type=Path, default=HERE / "request.t2i.json", help="JSON request body")
    parser.add_argument("--prompt", type=Path, help="prompt file; replaces the prompt in the request")
    parser.add_argument("--size", help="WIDTHxHEIGHT; replaces the size in the request")
    parser.add_argument("--out", default="t2i.png")
    args = parser.parse_args()

    body = json.loads(args.request.read_text())
    if args.prompt:
        body["prompt"] = args.prompt.read_text()
    if args.size:
        body["size"] = args.size

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.load(response)
    with open(args.out, "wb") as image:
        image.write(base64.b64decode(payload["data"][0]["b64_json"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

```


---

## FILE: examples/edit.py (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/edit.py)

```
#!/usr/bin/env python3
"""Send reference1.png and prompt.edit.txt to a MiniMax-H3-Image server and write the PNG.

    python3 edit.py [--url http://127.0.0.1:30000] [--out edit.png]

The images are sent in --image order, which defines <Picture 1>, <Picture 2>, ... in the
prompt. Uses only the Python standard library.
"""

import argparse
import base64
import json
import mimetypes
import urllib.request
import uuid
from pathlib import Path

# absolute(), not resolve(): in the Hugging Face cache this file is a symlink into blobs/.
HERE = Path(__file__).absolute().parent


def multipart(fields: dict, images: list) -> tuple:
    boundary = f"minimax-h3-image-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for path in images:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30000", help="server base URL")
    parser.add_argument("--image", action="append", type=Path,
                        help="reference image; repeat for <Picture 2>, ... (default: reference1.png)")
    parser.add_argument("--prompt", type=Path, default=HERE / "prompt.edit.txt", help="prompt file")
    parser.add_argument("--size", default="4096x4096", help="WIDTHxHEIGHT")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="edit.png")
    args = parser.parse_args()

    fields = {
        "prompt": args.prompt.read_text(),
        "size": args.size,
        "seed": str(args.seed),
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }
    body, content_type = multipart(fields, args.image or [HERE / "reference1.png"])
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/images/edits", data=body, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.load(response)
    with open(args.out, "wb") as image:
        image.write(base64.b64decode(payload["data"][0]["b64_json"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

```


---

## FILE: examples/api_generate.py (https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/HEAD/examples/api_generate.py)

```
#!/usr/bin/env python3
"""Generate the same image with the MiniMax API, as a reference for the local result.

    export MINIMAX_API_BASE=https://api.minimax.cn   # CN; Global: https://api.minimax.io
    export MINIMAX_API_KEY=<API key>
    python3 api_generate.py t2i    # POST /v1/images/generations with instruction.t2i.txt
    python3 api_generate.py edit   # POST /v1/images/edits with instruction.edit.txt and reference1.png

The API applies the same prompt enhancement internally. Writes the image (api_<task>.png) and
the prompt the API used (api_<task>_revised_prompt.txt). Uses only the Python standard library.
"""

import argparse
import base64
import json
from pathlib import Path

from context_ir import HERE, add_task_arguments, post, request_body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_arguments(parser)
    args = parser.parse_args()

    instruction = args.instruction or HERE / f"instruction.{args.task}.txt"
    body = request_body(args.task, instruction, args.image, args.size)
    body.update({"n": 1, "response_format": "b64_json", "output_format": "png"})
    path = "/v1/images/generations" if args.task == "t2i" else "/v1/images/edits"
    result = post(path, body)
    item = result["data"][0]
    if "error" in item:
        raise RuntimeError(f"image failed: {item['error']} (request_id {result.get('request_id')})")
    Path(f"api_{args.task}.png").write_bytes(base64.b64decode(item["b64_json"]))
    Path(f"api_{args.task}_revised_prompt.txt").write_text(item["revised_prompt"])
    print(json.dumps({"request_id": result.get("request_id"), "size": item.get("size"), "usage": result.get("usage")}))


if __name__ == "__main__":
    main()

```


---

# ISSUES / PRs of MiniMax-AI/MiniMax-H3-Image


### #9 [open] PR: Join the two clauses with a semicolon instead of a comma
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/9  by lunar-me 2026-11-20T15:16:18Z

## Summary

`vLLM-Omni's differ slightly` and `see the vLLM-Omni recipe` are two independent clauses joined by a comma alone — a comma splice. A semicolon separates them correctly and keeps the imperative.

## Change

```diff
- vLLM-Omni's differ slightly, see the
+ vLLM-Omni's differ slightly; see the
```

`skills/h3-image-prompt-writing/SKILL.md`, line 138.

## Scope

Documentation only, English prose. No code, no behaviour, and nothing in the meaning of the sentence changes — only its grammar.

---

Found while proofreading the documentation. Happy to adjust the wording if you would rather phrase it differently.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/enable-autofix?expires=1797779782&installation_model_id=427615&pr_number=9&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F9&signature=f455d750588956b6bc281bfc56a487f2873caf7c09c96fcb4a64db7c49169934"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/9?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #8 [open] PR: Complete the authorization bullet: article and closing period
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/8  by lunar-me 2026-11-20T15:16:13Z

## Summary

`through Application form` is missing the determiner the count noun needs, and this bullet alone ends without a period, unlike the two around it.

## Change

```diff
- is requested through [Application form](https://platform.minimax.io/console/license)
+ is requested through the [Application form](https://platform.minimax.io/console/license).
```

`README.md`, line 438.

## Scope

Documentation only, English prose. No code, no behaviour, and nothing in the meaning of the sentence changes — only its grammar.

---

Found while proofreading the documentation. Happy to adjust the wording if you would rather phrase it differently.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/enable-autofix?expires=1797779776&installation_model_id=427615&pr_number=8&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F8&signature=4464371b322dd5e61d4cfa56b7d37eb2de55a48586d0e836864415af775f528b"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/8?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #7 [open] PR: Add the missing article before the license name
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/7  by lunar-me 2026-11-20T15:16:08Z

## Summary

The license name is a count noun, so it takes a determiner after `subject to`. The bullet directly above writes `under the [MiniMax-H3-Image Community License Agreement]`, which makes the omission here inconsistent as well as ungrammatical.

## Change

```diff
- is accordingly subject to [MiniMax-H3-Image Community License Agreement](LICENSE).
+ is accordingly subject to the [MiniMax-H3-Image Community License Agreement](LICENSE).
```

`README.md`, line 437.

## Scope

Documentation only, English prose. No code, no behaviour, and nothing in the meaning of the sentence changes — only its grammar.

---

Found while proofreading the documentation. Happy to adjust the wording if you would rather phrase it differently.

<!-- codesmith:footer -->
---
<a href="https://dashboardbackend.blacksmith.sh/track/enable-autofix?expires=1797779773&installation_model_id=427615&pr_number=7&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F7&signature=739e28e9cd1037779642e537c201e8c41918d253a03cf64edcf8775a8d4cb544"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/7?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #6 [open] PR: Add h3metal (Apple Silicon) to the deployment list
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/6  by audenvenn 2026-11-12T16:48:21Z

h3metal runs MiniMax-H3-Image (text-to-image and editing with references) on Apple Silicon Macs, with its own Metal engine for the GPU matrix units of M5/M6-class chips. On a Mac mini (M6, 24 GB) a 2048x2048 image takes 41 s, 2.8x faster than h3-image-mlx on the same machine. Each part is tested against the diffusers and transformers modules; the benchmarks and the method are in the repo.

It converts the official checkpoint locally and keeps the model under the MiniMax-H3-Image Community License: its README states the territory and acceptable-use terms, and the PNGs it writes say they are AI-generated. It is a community project, not affiliated with MiniMax.

The line goes at the end of the list in all four READMEs, so it does not conflict with #5.

Thanks for releasing the weights.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1797094105&installation_model_id=427615&pr_number=6&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F6&signature=8ace21e67b0686ac4f2bcbae95732b53bea8339bce96c1051d33064d9b6fee82"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/6?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #5 [open] PR: Add LightX2V Deployment Guide
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/5  by helloyongyang 2026-11-12T10:25:52Z



<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1797071155&installation_model_id=427615&pr_number=5&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F5&signature=824736c80148f068f18bb5910db96a393e506997caa10d69c18ba9a66c36d886"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/5?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #4 [open] ISSUE: 能开源没蒸馏CFG的权重版本吗
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/issues/4  by bigcat001 2026-11-12T08:24:42Z




### #3 [closed] PR: Update the showcase and skills pages
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/3  by SaladDay 2026-11-11T18:29:24Z

Follows #2.

- **Showcase contents** (`showcase/README.md` and its zh-CN, ja and ko versions): a short intro, then one table with the guide's two categories and seven sections in its order, each with a cover image and a one-sentence description from the guide.
- **Section pages**: one page per section of the guide. Education, product design and robotics are now the three parts of one page, Precise Instruction Following and Understanding, as in the guide; Game Worlds has its two parts as subheadings. Each page opens with the section's description.
- **Skills page** (`skills/README.md`): laid out like the MiniMax-H3 skills page, with an image of three results rendered from prompts the skill wrote. Text that repeated the README is removed.

All anchors and relative links in the four READMEs, the 32 showcase pages and `skills/README.md` resolve.

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1797013768&installation_model_id=427615&pr_number=3&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F3&signature=31f902d177c635f2dec8ea44f36c507c505d2027fd42791cea170c46901894f6"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/3?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->


### #2 [closed] PR: Restructure README, add a showcase gallery and rewrite the prompt skill
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/2  by SaladDay 2026-11-11T12:54:16Z

Restructures the README into the GitHub-side usage guide, following the form of the MiniMax-H3 repository, adds a showcase gallery with the official guide's cases and prompts, and rewrites the bundled prompt writing skill. The Hugging Face model card stays the release page.

**Structure**

| Chapter | Parts |
| --- | --- |
| Header | Centered title, tagline *One idea with open weights. A world of uses.*, navigation |
| Prompt Writing Skill | Install, example request, what rendering needs |
| Showcase | A grid of 68 cases from the official user guide, in its two categories and order (Advanced Open-Source Capabilities 20, Creative Applications 48, four per row). Each tile opens the case on its section page in `showcase/` |
| Use Online | Web app, desktop app, API platform and API reference, Global and China |
| Overview | Positioning in three lines, Specifications (official parameter table, 4K marked as best), Model Zoo, Model Architecture |
| Run Locally | Download, Deployment (SGLang, vLLM-Omni), Request Parameters, Sizes, Full Workflow |
| Prompting Tips | Five tips from the official user guide |
| License, Contact Us | |

**What changes**
- **README**: each point is stated once; English is 442 lines, 110 of them the showcase grid, down from 875 in an earlier revision of this PR (329 on `main`), with the same chapters in `README.zh-CN.md`, `README.ja.md` and `README.ko.md`.
- **Showcase pages**: `showcase/` has one page per section of the guide in each language and a contents page, `showcase/README.md`, which GitHub also shows when the folder is opened. The pages show each image larger, the input images of an edit next to its output, and the prompt in a folded block: the official English guide for en/ja/ko (nine prompts the English guide lacks are translated from the Chinese), the Chinese originals for zh-CN. The three animated results from the guide play on these pages. GitHub does not lazy-load images in Markdown, so each page downloads only its own section, 1 to 8 MB. The full-width composites in `assets/showcase/` are replaced by one file per image, sized for the page, and 480x360 tiles.
- **Cases left out**: 10 of the guide's 78. Two have prompts that do not match their image; the others show a real game title, a real designer's name, a journal masthead with a garbled license line, a third-party artwork as input, a misspelled headline, or a likeness of an existing character.
- **Prompt writing skill**: `SKILL.md` is rewritten around the official prompting tips, with a workflow, writing rules, an output format with the aspect ratio and a final check. The new `scripts/generate.py` renders the prompt through the MiniMax API (standard library only). The text-to-image guide no longer turns spaces in quoted text into full-width spaces, which rendered as visibly wide gaps.
- **Run Locally**: scoped download (about 135 GB instead of 204 GB), SGLang and vLLM-Omni commands from the SGLang cookbook and vLLM-Omni recipe, including one H200, a size rule for other aspect ratios, and an agent alternative for prompt enhancement in Full Workflow.

**Checked**
- `generate.py` was run against the MiniMax API for text-to-image (1K and 4K, `n` up to 2), editing with PNG and WebP references, and editing with an aspect ratio, plus its error paths.
- The skill was tested on five briefs (poster with Chinese text, infographic, photo, game concept, bar chart). Prompts written with it and rendered without API expansion kept the requested text exactly and added no unrequested text.
- All anchors and relative links in the four READMEs, the 40 showcase pages and `skills/README.md` resolve. The 68 tile links per language match the heading ids that GitHub's renderer generates, and the folded prompts survive its sanitizer. The pages were reviewed rendered with GitHub's own stylesheet.



### #1 [closed] PR: Fix README and LICENSE links
URL: https://github.com/MiniMax-AI/MiniMax-H3-Image/pull/1  by liddalidd 2026-11-11T07:46:49Z

Link-only changes to README.md and LICENSE that bring their links in line with the Hugging Face model card (huggingface.co/MiniMaxAI/MiniMax-H3-Image). No other content changes.

README.md
- GitHub badge: MiniMax-AI/MiniMax-H3 -> MiniMax-AI/MiniMax-H3-Image (this repository).
- ModelScope badge: organization page -> model page modelscope.cn/models/MiniMax/MiniMax-H3-Image.
- WeChat badge: platform.minimaxi.com/docs/faq/contact-us -> platform.minimax.cn/docs/faq/contact-us, which the old URL redirects to.
- SGLang cookbook (2 places): .../MiniMax/MiniMax-H3-image -> .../MiniMax/MiniMax-H3-Image; the lowercase URL only works through a redirect.
- vLLM: linked to vllm-project/vllm with an empty "vllm recipes" link. The model is served by vLLM-Omni, so it now links to vllm-project/vllm-omni and the recipe at recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image.
- diffusers docs, Comfy tutorial and the ComfyUI T2I / Edit templates: the links were empty and are now filled in.
- License section: platform.minimax.io/license returns 404. The sentence now links to the application form (platform.minimax.io/console/license), and the separate "Application form" bullet is merged into it.

LICENSE
- Section IV.1: platform.minimax.io/license (404) -> platform.minimax.io/console/license. LICENSE is now identical to the copy on Hugging Face.

All new link targets return HTTP 200.

<!-- codesmith:footer -->
---
<a href="https://backend.blacksmith.sh/track/enable-autofix?expires=1796975213&installation_model_id=427615&pr_number=1&ref=codesmith_pr_footer&repository=MiniMax-AI%2FMiniMax-H3-Image&return_to=https%3A%2F%2Fgithub.com%2FMiniMax-AI%2FMiniMax-H3-Image%2Fpull%2F1&signature=9e297b1ce163107bea1c0437c87ec65162bc788234ca69dd82d51fc645afa4c3"><picture><source media="(prefers-color-scheme: dark)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-light.svg"><img alt="Autofix with [code]smith" src="https://pr-comments-assets.blacksmith.sh/codesmith/autofix-with-codesmith-dark-3x.png" width="190" height="28"></picture></a>
<sup><a href="https://app.blacksmith.sh/MiniMax-AI/codesmith/MiniMax-H3-Image/pr/1?autoLogin=true&ref=codesmith_pr_footer">Open in [code]smith</a> · Need help on this PR? Tag <code>@codesmith-bot</code> with what you need.</sup>

<!-- codesmith:autofix:disabled -->
<!-- /codesmith:footer -->
