import os
import zipfile
import time
import joblib



def getstrtime():
    output_file=time.strftime("%m%d%H%M", time.localtime())
    return output_file

def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()
 
import shutil
def save(t1_valid, t1_test, t2_valid, t2_test, save_path='result/temp', remove_oral_file=True):
    '''
     submission.zip
        ├── t1
            ├── test_pred.tsv     /* scores of test items */
            └── valid_pred.tsv    /* scores of validation items */
        ├── t2
            ├── test_pred.tsv     /* scores of test items */
            └── valid_pred.tsv    /* scores of validation items */
    '''
    t1_save_path = os.path.join(save_path, 't1')
    t2_save_path = os.path.join(save_path, 't2')
    if not os.path.exists(t1_save_path):
        os.makedirs(t1_save_path)
    if not os.path.exists(t2_save_path):
        os.makedirs(t2_save_path)
    
    t1_valid.to_csv(os.path.join(t1_save_path, 'valid_pred.tsv'), sep='\t', index=None)
    t2_valid.to_csv(os.path.join(t2_save_path, 'valid_pred.tsv'), sep='\t', index=None)
    t1_test.to_csv(os.path.join(t1_save_path, 'test_pred.tsv'), sep='\t', index=None)
    t2_test.to_csv(os.path.join(t2_save_path, 'test_pred.tsv'), sep='\t', index=None)
    
    # zip 
    save_zip_fn = os.path.join(save_path, 'submission.zip')
    # zipDir(save_path, save_zip_fn)

    # remove 
    # if remove_oral_file:
    #     shutil.rmtree(t1_save_path)
    #     shutil.rmtree(t2_save_path)
        # shutil.rmtree(save_path)
        
    
if __name__ == "__main__":
    input_path = "./origin_file_001"
    output_path = "./test.zip"
 
    zipDir(input_path, output_path)