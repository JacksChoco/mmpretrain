import splitfolders

input_folder = "/Users/ljs8860/github/mmpretrain/Teeth/Normal"
output = "/Users/ljs8860/github/mmpretrain/Normal-split/" 

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.