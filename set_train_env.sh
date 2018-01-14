# clone/clear F.LF
git clone https://github.com/Project-F/F.LF.git
cd F.LF
echo 'clean F.LF/'
git checkout .
git clean -fd
cd ..
# clone/clear LF2_19
git clone https://github.com/Project-F/LF2_19.git
cd LF2_19
echo 'clean LF2_19/'
git checkout .
git clean -fd
cd ..

# Modify Core
cp -f LF2_agent/replace_train/game.html F.LF/game/game.html
cp -f LF2_agent/replace_train/manager.js F.LF/LF/manager.js
cp -f LF2_agent/replace_train/global.js F.LF/LF/global.js
cp -f LF2_agent/replace_train/data.js LF2_19/data/data.js
cp -f LF2_agent/replace_train/DeepFighter.js LF2_19/AI/DeepFighter.js
# New Character
cp -f LF2_agent/replace_train/bandit_blue.js LF2_19/data/bandit_blue.js
cp -f LF2_agent/replace_train/properties.js LF2_19/data/properties.js
cp -f LF2_agent/replace_train/sprite/* LF2_19/sprite/