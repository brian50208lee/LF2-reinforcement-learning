# clone/clear F.LF
bash set_depedency.sh
# Modify Core
cp -f LF2_agent/replace_train/game.html F.LF/game/game.html
cp -f LF2_agent/replace_train/manager.js F.LF/LF/manager.js
cp -f LF2_agent/replace_test_pk/global.js F.LF/LF/global.js
cp -f LF2_agent/replace_train/data.js LF2_19/data/data.js
cp -f LF2_agent/replace_test_pk/DeepFighter.js LF2_19/AI/DeepFighter.js
# New Character
cp -f LF2_agent/replace_train/bandit_blue.js LF2_19/data/bandit_blue.js
cp -f LF2_agent/replace_train/properties.js LF2_19/data/properties.js
cp -f LF2_agent/replace_train/sprite/* LF2_19/sprite/