###################
# This script is modified based on: https://github.com/sindhura97/STraTS
# 提取入住过icu的成年患者
###################


import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np

# mimic_data_dir = 'path/to/mimic-iv-2.2/'
mimic_data_dir = '../../../Data/mimic-iv-2.2/'

# 入住所有重症监护室。loc为过滤，去除没有intime和outtime的数据。
icu = pd.read_csv(mimic_data_dir+'icustays.csv.gz', usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
icu.columns = icu.columns.str.upper()
icu.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
print(icu.columns)
icu = icu.loc[icu.INTIME.notna()]
icu = icu.loc[icu.OUTTIME.notna()]

# 过滤掉儿科患者。
pat = pd.read_csv(mimic_data_dir+'patients.csv.gz', usecols=['subject_id', 'anchor_year', 'anchor_age', 'dod', 'gender'])
pat.columns = pat.columns.str.upper()
pat.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
icu = icu.merge(pat, on='SUBJECT_ID', how='left')#以subject_id为连接键进行左连接
icu['INTIME'] = pd.to_datetime(icu.INTIME)
icu['AGE'] = icu.INTIME.map(lambda x: x.year) - icu['ANCHOR_YEAR'] + icu['ANCHOR_AGE']
icu = icu.loc[icu.AGE>=18]#任何与NaN进行比较的结果都会是False

#只保留ARDS患者
diag = pd.read_csv(mimic_data_dir+'diagnoses_icd.csv.gz', usecols=['subject_id', 'hadm_id', 'icd_code'])
diag.columns = diag.columns.str.upper()
ARDS = ['J80', '51882']
diag = diag.loc[diag.ICD_CODE.isin(ARDS)]
icu = icu.loc[icu.HADM_ID.isin(diag.HADM_ID)]

# Observation
# 提取icu住院的图表事件。
ch = []
for chunk in tqdm(pd.read_csv(mimic_data_dir+'chartevents.csv.gz', chunksize=10000000,
                usecols = ['hadm_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'])):
    chunk = chunk.loc[chunk.stay_id.isin(icu.ICUSTAY_ID)]
    chunk = chunk.loc[chunk.charttime.notna()]
    ch.append(chunk)
del chunk
print ('Done')
ch = pd.concat(ch)
ch.columns = ch.columns.str.upper()
ch.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
print ('Done')
ch = ch.loc[~(ch.VALUE.isna() & ch.VALUENUM.isna())]
ch['TABLE'] = 'chart'
print ('Done')

# 提取实验室事件。
la = pd.read_csv(mimic_data_dir+'labevents.csv.gz', usecols = ['hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom'])
la.columns = la.columns.str.upper()
la.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
la = la.loc[la.HADM_ID.isin(icu.HADM_ID)]
la.HADM_ID = la.HADM_ID.astype(int)
la = la.loc[la.CHARTTIME.notna()]
la = la.loc[~(la.VALUE.isna() & la.VALUENUM.isna())]#~取反，&与
la['ICUSTAY_ID'] = np.nan
la['TABLE'] = 'lab'
print("图表事件，实验室事件提取完成")

# 血压
print("blood pressure")
# 提取bp事件。删除异常值。
dbp = [220051, 225310,  220180,  224643, 227242]
sbp = [220050, 225309,  220179,  224167, 227243]
mbp = [220052, 225312,  224322,  220181]
ch_bp = ch.loc[ch.ITEMID.isin(dbp+sbp+mbp)]
ch_bp = ch_bp.loc[(ch_bp.VALUENUM>=0)&(ch_bp.VALUENUM<=375)]
ch_bp.loc[ch_bp.ITEMID.isin(dbp), 'NAME'] = 'DBP'
ch_bp.loc[ch_bp.ITEMID.isin(sbp), 'NAME'] = 'SBP'
ch_bp.loc[ch_bp.ITEMID.isin(mbp), 'NAME'] = 'MBP'
ch_bp['VALUEUOM'] = 'mmHg'
ch_bp['VALUE'] = None
events = ch_bp.copy()
del ch_bp

# 提取GCS事件。检查异常值.
print("GCS")
gcs_eye = [220739]
gcs_motor = [223901]
gcs_verbal = [223900]
ch_gcs = ch.loc[ch.ITEMID.isin(gcs_eye+gcs_motor+gcs_verbal)]
ch_gcs.loc[ch_gcs.ITEMID.isin(gcs_eye+gcs_motor+gcs_verbal), 'NAME'] = 'GCS'
ch_gcs['VALUEUOM'] = None
ch_gcs['VALUE'] = None
events = pd.concat([events, ch_gcs])
del ch_gcs

# 提取heart_rate事件。删除异常值.
print("heart_rate")
hr = [220045]
ch_hr = ch.loc[ch.ITEMID.isin(hr)]
ch_hr = ch_hr.loc[(ch_hr.VALUENUM>=0)&(ch_hr.VALUENUM<=390)]
ch_hr['NAME'] = 'HR'
ch_hr['VALUEUOM'] = 'bpm'
ch_hr['VALUE'] = None
events = pd.concat([events, ch_hr])
del ch_hr

# 提取呼吸率事件。删除异常值。检查单元一致性。
print("respiratory_rate")
rr = [220210,  224689,  224422,  224690,  224688, 227860, 227918]
ch_rr = ch.loc[ch.ITEMID.isin(rr)]
ch_rr = ch_rr.loc[(ch_rr.VALUENUM>=0)&(ch_rr.VALUENUM<=330)]
ch_rr['NAME'] = 'RR'
ch_rr['VALUEUOM'] = 'brpm'
ch_rr['VALUE'] = None
events = pd.concat([events, ch_rr])
del ch_rr

# 提取温度事件。将F转换为C。删除异常值。
print("temperature")
temp_c = [223762]
temp_f = [223761]
ch_temp_c = ch.loc[ch.ITEMID.isin(temp_c)]
ch_temp_f = ch.loc[ch.ITEMID.isin(temp_f)]
ch_temp_f.VALUENUM = (ch_temp_f.VALUENUM-32)*5/9
ch_temp = pd.concat([ch_temp_c, ch_temp_f])
del ch_temp_c
del ch_temp_f
ch_temp = ch_temp.loc[(ch_temp.VALUENUM>=14.2)&(ch_temp.VALUENUM<=47)]
ch_temp['NAME'] = 'Temperature'
ch_temp['VALUEUOM'] = 'C'
ch_temp['VALUE'] = None
events = pd.concat([events, ch_temp])
del ch_temp


# 提取fio2事件。将%转换为分数。删除异常值。
print("fio2")
fio2 = [223835]
ch_fio2 = ch.loc[ch.ITEMID.isin(fio2)]
idx = ch_fio2.VALUENUM>1.0
ch_fio2.loc[idx, 'VALUENUM'] = ch_fio2.loc[idx, 'VALUENUM'] / 100
ch_fio2 = ch_fio2.loc[(ch_fio2.VALUENUM>=0.2)&(ch_fio2.VALUENUM<=1)]
ch_fio2['NAME'] = 'FiO2'
ch_fio2['VALUEUOM'] = None
ch_fio2['VALUE'] = None
events = pd.concat([events, ch_fio2])
del ch_fio2

# 提取毛细管再填充速率事件。转换为二进制。
print("capillary refill rate")
cr = [224308, 223951]
ch_cr = ch.loc[ch.ITEMID.isin(cr)]
ch_cr = ch_cr.loc[~(ch_cr.VALUE=='Other/Remarks')]
idx = (ch_cr.VALUE=='Normal <3 Seconds')|(ch_cr.VALUE=='Normal <3 secs')
ch_cr.loc[idx, 'VALUENUM'] = 1
idx = (ch_cr.VALUE=='Abnormal >3 Seconds')|(ch_cr.VALUE=='Abnormal >3 secs')
ch_cr.loc[idx, 'VALUENUM'] = 2
ch_cr['VALUEUOM'] = None
ch_cr['NAME'] = 'CRR'
events = pd.concat([events, ch_cr])
del ch_cr

# 提取葡萄糖事件。删除异常值。
print("glucose")
gl = [225664,220621,226537]
ev_gl = ch.loc[ch.ITEMID.isin(gl)]
ev_gl = ev_gl.loc[(ev_gl.VALUENUM>=0)&(ev_gl.VALUENUM<=2200)]
ev_gl['NAME'] = 'Glucose'
ev_gl['VALUEUOM'] = 'mg/dL'
ev_gl['VALUE'] = None
events = pd.concat([events, ev_gl])
del ev_gl

# 提取胆红素事件。删除异常值。
print("bilirubin")
br_to = [50885]
ev_br = la.loc[la.ITEMID.isin(br_to)]
ev_br = ev_br.loc[(ev_br.VALUENUM>=0)&(ev_br.VALUENUM<=66)]
ev_br.loc[ev_br.ITEMID.isin(br_to), 'NAME'] = 'Bilirubin'
ev_br['VALUEUOM'] = 'mg/dL'
ev_br['VALUE'] = None
events = pd.concat([events, ev_br])
del ev_br

# 提取插管事件。
print("intubated")
itb = [50812]
la_itb = la.loc[la.ITEMID.isin(itb)]
idx = (la_itb.VALUE=='INTUBATED')
la_itb.loc[idx, 'VALUENUM'] = 1
idx = (la_itb.VALUE=='NOT INTUBATED')
la_itb.loc[idx, 'VALUENUM'] = 0
la_itb['VALUEUOM'] = None
la_itb['NAME'] = 'Intubated'
events = pd.concat([events, la_itb])
del la_itb

# 提取多个事件。删除异常值。
print("multiple events")
o2sat = [50817, 220227, 220277]
sod = [50983, 50824]
pot = [50971, 50822]
mg = [50960]
po4 = [50970]
ca_total = [50893]
ca_free = [50808]
wbc = [51301, 51300]
hct = [50810, 51221]
hgb = [51222, 50811]
cl = [50902, 50806]
bic = [50882, 50803]
alt = [50861]
alp = [50863]
ast = [50878]
alb = [50862]
lac = [50813]
ld = [50954]
usg = [51498]
ph_ur = [51491, 51094, 220734]
ph_bl = [50820]
po2 = [50821]
pco2 = [50818]
tco2 = [50804]
be = [50802]
monos = [51254]
baso = [51146]
eos = [51200]
neuts = [51256]
lym_per = [51244, 51245]
lym_abs = [51133]
pt = [51274]
ptt = [51275]
inr = [51237]
agap = [50868]
bun = [51006]
cr_bl = [50912]
cr_ur = [51082]
mch = [51248]
mchc = [51249]
mcv = [51250]
rdw = [51277]
plt = [51265]
rbc = [51279]

# 44
features = {'O2 Saturation': [o2sat, [0,100], '%'],
            'Sodium': [sod, [0,250], 'mEq/L'], 
            'Potassium': [pot, [0,15], 'mEq/L'], 
            'Magnesium': [mg, [0,22], 'mg/dL'], 
            'Phosphate': [po4, [0,22], 'mg/dL'],
            'Calcium Total': [ca_total, [0,40], 'mg/dL'],
            'Calcium Free': [ca_free, [0,10], 'mmol/L'],
            'WBC': [wbc, [0,1100], 'K/uL'], 
            'Hct': [hct, [0,100], '%'], 
            'Hgb': [hgb, [0,30], 'g/dL'], 
            'Chloride': [cl, [0,200], 'mEq/L'],
            'Bicarbonate': [bic, [0,66], 'mEq/L'],
            'ALT': [alt, [0,11000], 'IU/L'],
            'ALP': [alp, [0,4000], 'IU/L'],
            'AST': [ast, [0,22000], 'IU/L'],
            'Albumin': [alb, [0,10], 'g/dL'],
            'Lactate': [lac, [0,33], 'mmol/L'],
            'LDH': [ld, [0,35000], 'IU/L'],
            'SG Urine': [usg, [0,2], ''],
            'pH Urine': [ph_ur, [0,14], ''],
            'pH Blood': [ph_bl, [0,14], ''],
            'PO2': [po2, [0,770], 'mmHg'],
            'PCO2': [pco2, [0,220], 'mmHg'],
            'Total CO2': [tco2, [0,65], 'mEq/L'],
            'Base Excess': [be, [-31, 28], 'mEq/L'],
            'Monocytes': [monos, [0,100], '%'],
            'Basophils': [baso, [0,100], '%'],
            'Eoisinophils': [eos, [0,100], '%'],
            'Neutrophils': [neuts, [0,100], '%'],
            'Lymphocytes': [lym_per, [0,100], '%'],
            'Lymphocytes (Absolute)': [lym_abs, [0,25000], '#/uL'],
            'PT': [pt, [0,150], 'sec'],
            'PTT': [ptt, [0,150], 'sec'],
            'INR': [inr, [0,150], ''],
            'Anion Gap': [agap, [0,55], 'mg/dL'],
            'BUN': [bun, [0,275], 'mEq/L'],
            'Creatinine Blood': [cr_bl, [0,66], 'mg/dL'],
            'Creatinine Urine': [cr_ur, [0,650], 'mg/dL'],
            'MCH': [mch, [0,50], 'pg'],
            'MCHC': [mchc, [0,50], '%'],
            'MCV': [mcv, [0,150], 'fL'],
            'RDW': [rdw, [0,37], '%'],
            'Platelet Count': [plt, [0,2200], 'K/uL'],
            'RBC': [rbc, [0,14], 'm/uL']
            }

for k, v in features.items():
    print (k)
    ev_k = pd.concat((ch.loc[ch.ITEMID.isin(v[0])], la.loc[la.ITEMID.isin(v[0])]))
    ev_k = ev_k.loc[(ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])]
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    ev_k['VALUE'] = None
    assert (ev_k.VALUENUM.isna().sum()==0)
    events = pd.concat([events, ev_k])
del ev_k

# 抽取机械通风。转换为二进制。
print("mechanical ventilation")
vent = [223848, 223849, 224687, 224685, 224684, 224686, 224697, 224695, 224696, 224746, 224747, 226873, 224738, 224419, 224750, 227187, 224707, 224709, 224705, 224706, 220339, 224700, 224702, 224701]
ch_vent = ch.loc[ch.ITEMID.isin(vent)]
ch_vent = ch_vent.loc[~(ch_vent.VALUE=='Other/Remarks')|~(ch_vent.VALUE=='Other')]
ch_vent['VALUENUM'] = 1
ch_vent['NAME'] = 'Mechanical Ventilation'
ch_vent['VALUEUOM'] = None
ch_vent['VALUE'] = None
events = pd.concat([events, ch_vent])
del ch_vent

# 释放内存。
del ch, la

# 提取输出事件。
oe = pd.read_csv(mimic_data_dir+'outputevents.csv.gz', usecols = ['stay_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom'])
oe.columns = oe.columns.str.upper()
oe.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
oe = oe.loc[oe.VALUE.notna()]
oe['VALUENUM'] = oe.VALUE
oe.VALUE = None
oe = oe.loc[oe.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]
oe.ICUSTAY_ID = oe.ICUSTAY_ID.astype(int)
oe['TABLE'] = 'output'

# # 从D_items.csv.gz中提取有关输出项的信息。
items = pd.read_csv(mimic_data_dir+'d_items.csv.gz', usecols=['itemid', 'label', 'abbreviation', 'unitname', 'param_type'])
items.columns = items.columns.str.upper()
items.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
items.loc[items.LABEL.isna(), 'LABEL'] = ''
items.LABEL = items.LABEL.str.lower()
oeitems = oe[['ITEMID']].drop_duplicates()#oe中选择ITEMID列，并删除重复的ITEMID值
oeitems = oeitems.merge(items, on='ITEMID', how='left')
'''
uf = [40286]
print ('Ultrafiltrate')
ev_k = oe.loc[oe.ITEMID.isin(uf)]
ind = (ev_k.VALUENUM>=0)&(ev_k.VALUENUM<=7000)
med = ev_k.VALUENUM.loc[ind].median()
ev_k.loc[~ind, 'VALUENUM'] = med
ev_k['NAME'] = 'Ultrafiltrate'
ev_k['VALUEUOM'] = 'mL'
events = pd.concat([events, ev_k])
del ev_k
'''
# # 提取多个事件。将异常值替换为中值。
print("multiple output events")
keys = ['urine', 'foley', 'void', 'nephrostomy', 'condom', 'drainage bag']
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(axis='columns')
ur = list(oeitems.loc[cond].ITEMID)
keys = ['stool', 'fecal', 'colostomy', 'ileostomy', 'rectal']
cond = pd.concat([oeitems.LABEL.str.contains(k) for k in keys], axis=1).any(axis='columns')
st = list(oeitems.loc[cond].ITEMID)
ct = list(oeitems.loc[oeitems.LABEL.str.contains('chest tube')].ITEMID) + [226593, 226590, 226591, 226595, 226592]
gs = [226576, 226575, 226573, 226630]
ebl = [226626, 226629]
em = [226571]
jp = list(oeitems.loc[oeitems.LABEL.str.contains('jackson')].ITEMID)
res = [227510, 227511]
pre = [226633]

# # 10
features = {'Urine': [ur, [0,2500], 'mL'],
            'Stool': [st, [0,4000], 'mL'],
            'Chest Tube': [ct, [0,2500], 'mL'],
            'Gastric': [gs, [0,4000], 'mL'],
            'EBL': [ebl, [0,10000], 'mL'],
#             'Pre-admission': [pre, [0,13000], 'mL'], # 错误地重复。
            'Emesis': [em, [0,2000], 'mL'],
            'Jackson-Pratt': [jp, [0,2000], 'ml'],
            'Residual': [res, [0, 1050], 'mL'],
            'Pre-admission Output': [pre, [0, 13000], 'ml']
            }

for k, v in features.items():
    print (k)
    ev_k = oe.loc[oe.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, 'VALUENUM'] = med
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    events = pd.concat([events, ev_k])
del ev_k


# 提取输入事件。
ie = pd.read_csv(mimic_data_dir+'inputevents.csv.gz',
    usecols = ['stay_id', 'hadm_id', 'itemid', 'starttime', 'endtime',
               'amount', 'amountuom'])
ie.columns = ie.columns.str.upper()
ie.rename(columns={'STAY_ID':'ICUSTAY_ID'}, inplace=True)
ie = ie.loc[ie.ICUSTAY_ID.isin(icu.ICUSTAY_ID)]

# 每小时拆分间隔。
ie.STARTTIME = pd.to_datetime(ie.STARTTIME)
ie.ENDTIME = pd.to_datetime(ie.ENDTIME)
ie['TD'] = ie.ENDTIME - ie.STARTTIME
new_ie_mv = ie.loc[ie.TD<=pd.Timedelta(1,'h')].drop(columns=['STARTTIME', 'TD'])
ie = ie.loc[ie.TD>pd.Timedelta(1,'h')]
new_rows = []
for _,row in tqdm(ie.iterrows()):
    admid, icuid, iid, amo, uom, stm, td = row.HADM_ID, row.ICUSTAY_ID, row.ITEMID, row.AMOUNT, row.AMOUNTUOM, row.STARTTIME, row.TD
    td = td.total_seconds()/60
    num_hours = td // 60
    hour_amount = 60*amo/td
    for i in range(1,int(num_hours)+1):
        new_rows.append([admid, icuid, iid, stm+pd.Timedelta(i,'h'), hour_amount, uom])
    rem_mins = td % 60
    if rem_mins>0:
        new_rows.append([admid, icuid, iid, row['ENDTIME'], rem_mins*amo/td, uom])
new_rows = pd.DataFrame(new_rows, columns=['HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM'])
new_ie_mv = pd.concat((new_ie_mv, new_rows))
ie = new_ie_mv.copy()
del new_ie_mv
ie['TABLE'] = 'input_mv' 
ie.rename(columns={'ENDTIME':'CHARTTIME'}, inplace=True)
ie.rename(columns={'AMOUNT':'VALUENUM', 'AMOUNTUOM':'VALUEUOM'}, inplace=True)
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)

# Convert mcg->mg, L->ml.
ind = (ie.VALUEUOM=='mcg')
ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM']*0.001
ie.loc[ind, 'VALUEUOM'] = 'mg'
ind = (ie.VALUEUOM=='L')
ie.loc[ind, 'VALUENUM'] = ie.loc[ind, 'VALUENUM']*1000
ie.loc[ind, 'VALUEUOM'] = 'ml'

# 提取加压素事件。删除异常值。
print("Vasopressin")
vaso = [222315]
ev_vaso = ie.loc[ie.ITEMID.isin(vaso)]
ind1 = (ev_vaso.VALUENUM==0)
ind2 = ev_vaso.VALUEUOM.isin(['U','units'])
ind3 = (ev_vaso.VALUENUM>=0)&(ev_vaso.VALUENUM<=400)
ind = ((ind2&ind3)|ind1)
med = ev_vaso.VALUENUM.loc[ind].median()
ev_vaso.loc[~ind, 'VALUENUM'] = med
ev_vaso['VALUEUOM'] = 'units'
ev_vaso['NAME'] = 'Vasopressin'
events = pd.concat([events, ev_vaso])
del ev_vaso

# 提取万古霉素事件。将剂量g转换为mg。删除异常值。
print("Vancomycin")
vanc = [225798]
ev_vanc = ie.loc[ie.ITEMID.isin(vanc)]
ind = ev_vanc.VALUEUOM.isin(['mg'])
ev_vanc.loc[ind, 'VALUENUM'] = ev_vanc.loc[ind, 'VALUENUM']*0.001 
ev_vanc['VALUEUOM'] = 'g'
ind = (ev_vanc.VALUENUM>=0)&(ev_vanc.VALUENUM<=8)
med = ev_vanc.VALUENUM.loc[ind].median()
ev_vanc.loc[~ind, 'VALUENUM'] = med
ev_vanc['NAME'] = 'Vacomycin'
events = pd.concat([events, ev_vanc])
del ev_vanc

# 提取葡萄糖酸钙事件。转换单位。删除异常值。
print("Calcium Gluconate")
cagl = [221456, 227525]
ev_cagl = ie.loc[ie.ITEMID.isin(cagl)]
ind = ev_cagl.VALUEUOM.isin(['mg'])
ev_cagl.loc[ind, 'VALUENUM'] = ev_cagl.loc[ind, 'VALUENUM']*0.001 
ind1 = (ev_cagl.VALUENUM==0)
ind2 = ev_cagl.VALUEUOM.isin(['mg', 'gm', 'grams'])
ind3 = (ev_cagl.VALUENUM>=0)&(ev_cagl.VALUENUM<=200)
ind = (ind2&ind3)|ind1
med = ev_cagl.VALUENUM.loc[ind].median()
ev_cagl.loc[~ind, 'VALUENUM'] = med
ev_cagl['VALUEUOM'] = 'g'
ev_cagl['NAME'] = 'Calcium Gluconate'
events = pd.concat([events, ev_cagl])
del ev_cagl

# 提取呋塞米事件。删除异常值。
print("Furosemide")
furo = [221794, 228340]
ev_furo = ie.loc[ie.ITEMID.isin(furo)]
ind1 = (ev_furo.VALUENUM==0)
ind2 = (ev_furo.VALUEUOM=='mg')
ind3 = (ev_furo.VALUENUM>=0)&(ev_furo.VALUENUM<=250)
ind = ind1|(ind2&ind3)
med = ev_furo.VALUENUM.loc[ind].median()
ev_furo.loc[~ind, 'VALUENUM'] = med
ev_furo['VALUEUOM'] = 'mg'
ev_furo['NAME'] = 'Furosemide'
events = pd.concat([events, ev_furo])
del ev_furo

# 提取法莫替丁事件。删除异常值。
print("Famotidine")
famo = [225907]
ev_famo = ie.loc[ie.ITEMID.isin(famo)]
ind1 = (ev_famo.VALUENUM==0)
ind2 = (ev_famo.VALUEUOM=='dose')
ind3 = (ev_famo.VALUENUM>=0)&(ev_famo.VALUENUM<=1)
ind = ind1|(ind2&ind3)
med = ev_famo.VALUENUM.loc[ind].median()
ev_famo.loc[~ind, 'VALUENUM'] = med
ev_famo['VALUEUOM'] = 'dose'
ev_famo['NAME'] = 'Famotidine'
events = pd.concat([events, ev_famo])
del ev_famo

# 提取哌拉西林事件。转换单位。删除异常值。
print("Piperacillin")
pipe = [225893, 225892]
ev_pipe = ie.loc[ie.ITEMID.isin(pipe)]
ind1 = (ev_pipe.VALUENUM==0)
ind2 = (ev_pipe.VALUEUOM=='dose')
ind3 = (ev_pipe.VALUENUM>=0)&(ev_pipe.VALUENUM<=1)
ind = ind1|(ind2&ind3)
med = ev_pipe.VALUENUM.loc[ind].median()
ev_pipe.loc[~ind, 'VALUENUM'] = med
ev_pipe['VALUEUOM'] = 'dose'
ev_pipe['NAME'] = 'Piperacillin'
events = pd.concat([events, ev_pipe])
del ev_pipe

# 提取头孢唑林事件。转换单位。删除异常值。
print("Cefazolin")
cefa = [225850]
ev_cefa = ie.loc[ie.ITEMID.isin(cefa)]
ind1 = (ev_cefa.VALUENUM==0)
ind2 = (ev_cefa.VALUEUOM=='dose')
ind3 = (ev_cefa.VALUENUM>=0)&(ev_cefa.VALUENUM<=2)
ind = ind1|(ind2&ind3)
med = ev_cefa.VALUENUM.loc[ind].median()
ev_cefa.loc[~ind, 'VALUENUM'] = med 
ev_cefa['VALUEUOM'] = 'dose'
ev_cefa['NAME'] = 'Cefazolin'
events = pd.concat([events, ev_cefa])
del ev_cefa

# 提取光纤事件。删除异常值。
print("Fiber")
fibe = [225936, 227695, 225928, 226051, 226050, 226048,227699, 227696, 226049, 227698, 226027]
ev_fibe = ie.loc[ie.ITEMID.isin(fibe)]
ind1 = (ev_fibe.VALUENUM==0)
ind2 = (ev_fibe.VALUEUOM=='ml')
ind3 = (ev_fibe.VALUENUM>=0)&(ev_fibe.VALUENUM<=1600)
ind = ind1|(ind2&ind3)
med = ev_fibe.VALUENUM.loc[ind].median()
ev_fibe.loc[~ind, 'VALUENUM'] = med 
ev_fibe['NAME'] = 'Fiber'
ev_fibe['VALUEUOM'] = 'ml'
events = pd.concat([events, ev_fibe])
del ev_fibe

# 提取泮托拉唑事件。删除异常值。
print("Pantoprazole")
pant = [225910]
ev_pant = ie.loc[ie.ITEMID.isin(pant)]
ind = (ev_pant.VALUENUM>0)
ev_pant.loc[ind, 'VALUENUM'] = 1
ind = (ev_pant.VALUENUM>=0)
med = ev_pant.VALUENUM.loc[ind].median()
ev_pant.loc[~ind, 'VALUENUM'] = med
ev_pant['NAME'] = 'Pantoprazole'
ev_pant['VALUEUOM'] = 'dose'
events = pd.concat([events, ev_pant])
del ev_pant

# 提取硫酸镁事件。删除异常值。
print("Magnesium Sulphate")
masu = [222011, 227524]
ev_masu = ie.loc[ie.ITEMID.isin(masu)]
ind = (ev_masu.VALUEUOM=='mg')
ev_masu.loc[ind, 'VALUENUM'] = ev_masu.loc[ind, 'VALUENUM']*0.001
ind1 = (ev_masu.VALUENUM==0)
ind2 = ev_masu.VALUEUOM.isin(['gm', 'grams', 'mg'])
ind3 = (ev_masu.VALUENUM>=0)&(ev_masu.VALUENUM<=125)
ind = ind1|(ind2&ind3)
med = ev_masu.VALUENUM.loc[ind].median()
ev_masu.loc[~ind, 'VALUENUM'] = med 
ev_masu['VALUEUOM'] = 'g'
ev_masu['NAME'] = 'Magnesium Sulphate'
events = pd.concat([events, ev_masu])
del ev_masu

# 提取氯化钾事件。删除异常值。
print("Potassium Chloride")
poch = [225166, 227536]
ev_poch = ie.loc[ie.ITEMID.isin(poch)]
ind1 = (ev_poch.VALUENUM==0)
ind2 = ev_poch.VALUEUOM.isin(['mEq', 'mEq.'])
ind3 = (ev_poch.VALUENUM>=0)&(ev_poch.VALUENUM<=501)
ind = ind1|(ind2&ind3)
med = ev_poch.VALUENUM.loc[ind].median()
ev_poch.loc[~ind, 'VALUENUM'] = med 
ev_poch['VALUEUOM'] = 'mEq'
ev_poch['NAME'] = 'KCl'
events = pd.concat([events, ev_poch])
del ev_poch

# 提取多个事件。删除异常值。
print("multiple input events")
mida = [221668]
prop = [222168]
albu25 = [220862]
albu5 = [220864]
ffpl = [220970]
lora = [221385]
mosu = [225154]
game = [225799]
lari = [225828]
milr = [221986]
crys = [226364, 226375]
hepa = [225975, 225152]
prbc = [225168, 226368, 227070]
poin = [226452, 226377]
neos = [221749]
pigg = [226089]
nigl = [222056]
nipr = [222051]
meto = [225974]
nore = [221906]
coll = [226365, 226376]
hyzi = [221828]
gtfl = [226453]
hymo = [221833]
fent = [225942, 221744]
inre = [223258]
inhu = [223262]
ingl = [223260]
innp = [223259]
# nana = [30140]
d5wa = [220949]
doth = [225823,225825, 220950, 225827, 225941,220952,228142, 228140,228141]
nosa = [225158, 30018]
hans = [30020, 225159]
stwa = [225944, 30065]
frwa = [30058, 225797, 41430, 40872, 41915, 43936, 41619, 42429, 44492, 46169, 42554]
solu = [225943]
dopa = [30043, 221662]
epin = [30119, 221289, 30044]
amio = [30112, 221347, 228339, 45402]
tpnu = [30032, 225916, 225917, 30096]
msbo = [227523]
pcbo = [227522]
prad = [30054, 226361]

# 43
features = {'Midazolam': [mida, [0, 500], 'mg'],
            'Propofol': [prop, [0, 12000], 'mg'],
            'Albumin 25%': [albu25, [0, 750], 'ml'],
            'Albumin 5%': [albu5, [0, 1300], 'ml'],
            'Fresh Frozen Plasma': [ffpl, [0, 33000], 'ml'],
            'Lorazepam': [lora, [0, 300], 'mg'],
            'Morphine Sulfate': [mosu, [0, 4000], 'mg'],
            'Gastric Meds': [game, [0, 7000], 'ml'],
            'Lactated Ringers': [lari, [0, 17000], 'ml'],
            'Milrinone': [milr, [0, 50], 'ml'],
            'OR/PACU Crystalloid': [crys, [0, 22000], 'ml'],
            'Packed RBC': [prbc, [0, 17250], 'ml'],
            'PO intake': [poin, [0, 11000], 'ml'],
            'Neosynephrine': [neos, [0, 1200], 'mg'],
            'Piggyback': [pigg, [0, 1000], 'ml'],
            'Nitroglycerine': [nigl, [0, 350], 'mg'],
            'Nitroprusside': [nipr, [0, 430], 'mg'],
            'Metoprolol': [meto, [0, 151], 'mg'],
            'Norepinephrine': [nore, [0, 80], 'mg'],
            'Colloid': [coll, [0, 20000], 'ml'],
            'Hydralazine': [hyzi, [0, 80], 'mg'],
            'GT Flush': [gtfl, [0, 2100], 'ml'],
            'Hydromorphone': [hymo, [0, 125], 'mg'],
            'Fentanyl': [fent, [0, 20], 'mg'],
            'Insulin Regular': [inre, [0, 1500], 'units'],
            'Insulin Humalog': [inhu, [0, 340], 'units'],
            'Insulin largine': [ingl, [0, 150], 'units'],
            'Insulin NPH': [innp, [0, 100], 'units'],
            #'Unknown': [nana, [0, 1100], 'ml'],
            'D5W': [d5wa, [0,11000], 'ml'],
            'Dextrose Other': [doth, [0,4000], 'ml'],
            'Normal Saline': [nosa, [0, 11000], 'ml'],
            'Half Normal Saline': [hans, [0, 2000], 'ml'],
            'Sterile Water': [stwa, [0, 10000], 'ml'],
            'Free Water': [frwa, [0, 2500], 'ml'],
            'Solution': [solu, [0, 1500], 'ml'],
            'Dopamine': [dopa, [0, 1300], 'mg'],
            'Epinephrine': [epin, [0, 100], 'mg'],
            'Amiodarone': [amio, [0, 1200], 'mg'],
            'TPN': [tpnu, [0, 1600], 'ml'],
            'Magnesium Sulfate (Bolus)': [msbo, [0, 250], 'ml'],
            'KCl (Bolus)': [pcbo, [0, 500], 'ml'],
            'Pre-admission Intake': [prad, [0, 30000], 'ml']
            }

for k, v in features.items():
    print (k)
    ev_k = ie.loc[ie.ITEMID.isin(v[0])]
    ind = (ev_k.VALUENUM>=v[1][0])&(ev_k.VALUENUM<=v[1][1])
    med = ev_k.VALUENUM.loc[ind].median()
    ev_k.loc[~ind, 'VALUENUM'] = med
    ev_k['NAME'] = k
    ev_k['VALUEUOM'] = v[2]
    events = pd.concat([events, ev_k])
del ev_k

# 提取肝素事件。
print("heparin")
ev_k = ie.loc[ie.ITEMID.isin(hepa)]
ind1 = ev_k.VALUEUOM.isin(['U', 'units'])
ind2 = (ev_k.VALUENUM>=0)&(ev_k.VALUENUM<=25300)
ind = (ind1&ind2)
med = ev_k.VALUENUM.loc[ind].median()
ev_k.loc[~ind, 'VALUENUM'] = med
ev_k['NAME'] = 'Heparin'
ev_k['VALUEUOM'] = 'units'
events = pd.concat([events, ev_k])
del ev_k

# Save data.
print("saving done")
events.to_csv('mimic_ards_events.csv.gz', index=False)
icu.to_csv('mimic_ards_icu.csv.gz', index=False)
print("save done")
