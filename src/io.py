import os
import logging
import xlsxwriter

logger = logging.getLogger(__name__)


def save_rop_info(out_dir, property_values_list):
    # for backward compatibility
    c1_list = list(filter(lambda x: x['criterion'] == 'c1', property_values_list))
    c2_list = list(filter(lambda x: x['criterion'] == 'c2', property_values_list))
    c3_list = list(filter(lambda x: x['criterion'] == 'c3', property_values_list))

    c1_list.sort(key=lambda x: (x['reliability'], x['doi']), reverse=True)
    c2_list.sort(key=lambda x: (x['reliability'], x['doi']), reverse=True)
    c3_list.sort(key=lambda x: (x['reliability'], x['doi']), reverse=True)

    tgt_file = os.path.join(out_dir, 'ROP-Info.xlsx')

    # write header
    workbook = xlsxwriter.Workbook(tgt_file)
    worksheet = workbook.add_worksheet('Sheet 1')
    worksheet.write('A1', 'article idx')
    worksheet.write('B1', 'sentence idx')
    worksheet.write('C1', 'DOI')
    worksheet.write('D1', 'material')
    worksheet.write('E1', 'property mention')
    worksheet.write('F1', 'value')
    worksheet.write('G1', 'property type')
    worksheet.write('H1', 'sentence')
    worksheet.write('I1', 'local file')
    worksheet.write('J1', 'reliability')

    n_line = 2
    article_idx = 0

    prev_doi = ''
    for cat_list in [c1_list, c2_list, c3_list]:
        for rop_info_dict in cat_list:
            curr_doi = rop_info_dict['doi']
            if curr_doi != prev_doi:
                article_idx += 1
                prev_doi = curr_doi
            file_name = rop_info_dict['file-path']
            worksheet.write(f'A{n_line}', f'{article_idx}')
            worksheet.write(f'B{n_line}', f"{rop_info_dict['sentence-id']}")
            worksheet.write_url(f'C{n_line}', f'https://doi.org/{rop_info_dict["doi"]}', string=rop_info_dict["doi"])
            worksheet.write(f'D{n_line}', rop_info_dict['material'])
            worksheet.write(f'E{n_line}', rop_info_dict['property'])
            worksheet.write(f'F{n_line}', rop_info_dict['value'])
            worksheet.write(f'G{n_line}', rop_info_dict['type'])
            worksheet.write(f'H{n_line}', rop_info_dict['sentence'])
            worksheet.write_url(f'I{n_line}', f'external:{file_name}', string=os.path.basename(file_name))
            worksheet.write(f'J{n_line}', f"{rop_info_dict['reliability']: .1f}")
            n_line += 1

    workbook.close()
