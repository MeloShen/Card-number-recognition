# Card Number Recognition

This is a simple project that uses `OpenCV` to recognize credit card numbers. We need to input the Card image and template.
After reading the template, find the grayscale of the template and binary image, and then use `findContours` to find the outline of the template.

![template_reco](https://raw.githubusercontent.com/MeloShen/Card-number-recognition/main/_output/template-reco.png)

Then the target picture line operation, first the target picture line gray, and then use 'top hat' for the target picture line gray ha, eliminate interference, highlight details.



### License
[MIT](https://choosealicense.com/licenses/mit/)
