# -*- encoding: utf-8 -*-
import xlsxwriter
import base64
import glob
import os
import requests
requests.packages.urllib3.disable_warnings()
from collections import OrderedDict

res_dict = {
    "Name": [0, "姓名"],
    "Relationship": [1, "户主或与户主关系"],
    "Sex": [2, "性别"],
    "BirthAddress": [3, "出生地"],
    "Nation": [4, "民族"],
    "Birthday": [5, "生日"],
    "CardNo": [6, "身份证号"],
    "HouseholdNum": [7, "户号"],
    "FormerName": [8, "曾用名"],
    "Hometown": [9, "籍贯"],
    "OtherAddress": [10, "本市（县）其他住址"],
    "Belief": [11, "宗教信仰"],
    "Height": [12, "身高"],
    "BloodType": [13, "血型"],
    "Education": [14, "文化程度"],
    "MaritalStatus": [15, "婚姻状况"],
    "VeteranStatus": [16, "兵役状况"],
    "WorkAddress": [17, "服务处所"],
    "Career": [18, "职业"],
    "WWToCity": [19, "何时由何地迁来本市(县)"],
    "WWHere": [20, "何时由何地迁往本址"],
    "Date": [21, "登记日期"]
}


def pic_base64(image):
    with open(image, 'rb') as f:
        base64_data = base64.b64encode(f.read())
    return base64_data


class house_hold(object):

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_key = self.get_token_key()

    def get_token_key(self):
        url = f'https://aip.baidubce.com/oauth/2.0/token'
        res = requests.get(url, verify=False, params={
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret})
        token_content = res.json()
        assert "error" not in token_content, f"{token_content['error_description']}"
        token_key = token_content['access_token']
        return token_key

    def get_result(self, data):
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/household_register"
        params = dict()
        params['access_token'] = self.token_key
        res = requests.post(url=request_url,
                            data=data,
                            params=params,
                            verify=False,
                            headers={'Content-Type': 'application/x-www-form-urlencoded'})
        result = res.json()
        if 'error_code' in result:
            print('图片识别出错，请换图片进行识别！')
            return -1
        return result


def write_excel(excel_name, sheet_name, results):
    workbook = xlsxwriter.Workbook(excel_name)
    worksheet = workbook.add_worksheet(sheet_name)
    for i, key in enumerate(res_dict.keys()):
        worksheet.write(0, i, res_dict[key][1])

    for i, res in enumerate(results):
        for key in res.keys():
            worksheet.write(i + 1, res_dict[key][0], res[key]['words'])

    workbook.close()



def main():
    client_id = "替换为你自己的"
    client_secret = "替换为你自己的"
    img_folder = "images"
    images = glob.glob(os.path.join(img_folder, '*'))
    images.sort()
    results = list()
    data = dict()
    household_obj = house_hold(client_id, client_secret)
    for img in images:
        image_base64 = pic_base64(img)
        data['image'] = str(image_base64, encoding='utf-8')

        result = household_obj.get_result(data)
        if result == -1:
            continue
        results.append(result['words_result'])
    
    write_excel('house_hold.xlsx', '户口本信息', results)
    print('finish')



if __name__ == "__main__":
    main()
