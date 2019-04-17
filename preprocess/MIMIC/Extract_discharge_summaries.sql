select subject_id, hadm_id, string_agg(text , ' ' order by chartdate)
from mimiciii.noteevents 
where category = 'Discharge summary' 
and iserror is null
group by subject_id, hadm_id;