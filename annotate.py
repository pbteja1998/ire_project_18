import os,sys,re
import xml.dom.minidom

def parse_files(folder_path, out_folder_path):
	count = 1
	for file in os.listdir(folder_path):
		# print file
		f = open(os.path.join(out_folder_path, file), 'w+')
		doc = xml.dom.minidom.parse(os.path.join(folder_path, file))
		sen_list = doc.getElementsByTagName('A-S')
		for sen in sen_list:
			rhe_val = sen.getAttribute('AZ')
			sid = sen.getAttribute('ID')
			text = ''
			for node in sen.childNodes:
				if node.nodeType == node.TEXT_NODE:
					text += node.data.strip()
			if text == '':
				continue
			f.write(rhe_val + '\t' + text.replace('\n', ' ') + '\t' + sid +'\n')

		sen_list = doc.getElementsByTagName('S')
		for sen in sen_list:
			rhe_val = sen.getAttribute('AZ')
			sid = sen.getAttribute('ID')
			text = ''
			for node in sen.childNodes:
				if node.nodeType == node.TEXT_NODE:
					text += node.data.strip()
			if text == '':
				continue
			f.write(rhe_val + '\t' + text.replace('\n', ' ') + '\t' + sid +'\n')
		f.close()
		count += 1

if __name__ == '__main__':
	folder_path = sys.argv[1]
	out_folder_path = sys.argv[2]
	if not os.path.exists(out_folder_path):
		os.makedirs(out_folder_path)
	parse_files(folder_path, out_folder_path)