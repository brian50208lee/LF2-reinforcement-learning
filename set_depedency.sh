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