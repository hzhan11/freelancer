# freelancer
<br>
This repo is for case https://www.freelancer.cn/projects/c-programming/Write-Python-program-get-list<br>
<br>
1.Before run<br>
a) make sure you correctly install tesseract to your OS<br>
b) install cv2, pytesseract and related packages to your python env<br>
c) copy new/Penitentiary.traineddata to your tessdata folder. For MacOS as below:<br>
cp Penitentiary.traineddata /usr/local/share/tessdata/<br>
d) convert your pdf to png, for example put them under new/sample_drawing.png<br>
<br>
2.How to run<br>
a) run cmd like below<br>
python main.py new/sample_drawing.png > result.csv<br>
b) open your csv as below<br>
total 12 tags found as below<br>
GT,0380<br>
HSS,0302A<br>
VMT,0302<br>
ZSMH,0307A<br>
ZSH,Q3D7A<br>
SSL,0302<br>
VDS,0302<br>
VQI,0302<br>
SAL,0302<br>
HIC,0307A<br>
ZAHH,0307A<br>
ZAH,0307A<br>
<br>
3.Limitations<br>
Hoping receive following more information to achieve higher accuracy<br>
a)So far, font is supposed to be PenitentiaryFill, some chars are similar and hard to identified<br>
b)More samples and hints for the rules of tags, may generate higher accuracy<br>