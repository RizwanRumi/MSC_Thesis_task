import os

Source_Path = ".\Datasets\source\images"
Destination = '.\Datasets\destination\images'

os.mkdir(Destination)
folder_files = os.listdir(Source_Path)

"""
Example 1: 
    input: 2022-09-28_15-43-33.jpg
    output: 1_broken.jpg (change as folder name ex: healthy, broken, healthy_col_neg, broken_gray_neg)
    
Example 2:
    input: 1_broken_jpg.rf.f761af66a0f2e333cd3011a19429aab6.jpg
    output: 1_broken_f761af66a0f2e333cd3011a19429aab6.jpg
"""

def main():
    for count, filename in enumerate(folder_files):
        #example 1:
        # rename = str(count + 1) + "_broken" + ".jpg"

        #example 2:
        file = filename.split(".", 1)
        name = file[0].split("jpg",1)
        file_format = file[1].split(".", 1)
        
        rename = name[0]+file_format[1]
        print(rename)

        # rename the file name
        os.rename(os.path.join(Source_Path, filename), os.path.join(Destination, rename))


# Driver Code
if __name__ == '__main__':
    main()
