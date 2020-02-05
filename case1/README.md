# freelancer

1.Before run
a) make sure you correctly install tesseract to your OS
b) install cv2, pytesseract and related packages to your python env
c) copy new/Penitentiary.traineddata to your tessdata folder. For MacOS as below:
cp Penitentiary.traineddata /usr/local/share/tessdata/
d) convert your pdf to png, for example put them under new/sample_drawing.png

2.How to run
a) run cmd like below
python main.py new/sample_drawing.png > result.csv
b) open your csv as below
total 12 tags found as below
GT,0380
HSS,0302A
VMT,0302
ZSMH,0307A
ZSH,Q3D7A
SSL,0302
VDS,0302
VQI,0302
SAL,0302
HIC,0307A
ZAHH,0307A
ZAH,0307A

3.Limitations
Hoping receive following more information to achieve higher accuracy
a)So far, font is supposed to be PenitentiaryFill, some chars are similar and hard to identified
b)More samples and hints for the rules of tags, may generate higher accuracy