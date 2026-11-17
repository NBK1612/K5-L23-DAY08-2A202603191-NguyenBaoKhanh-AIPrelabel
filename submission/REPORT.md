# Báo cáo Day 08 — 2A202603191 - Nguyễn Bảo Khánh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp                                              | Model + cỡ                                                                  | Ngưỡng                                                                                                      | Vì sao (dẫn `run_id` trong `model_log.csv`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`, imgsz 1024, GPU)                                     | person 0.15 · bicycle 0.2 · motorbike 0.5 · car 0.5 · bus, truck mặc định 0.3 (từ `qc/threshold_sweep.csv`) | Detector COCO bắt tốt người và xe máy gần camera (r01: img_007 người lái 0.92, xe máy 0.84). Chỉ bicycle đạt recall ≥ 90% (0.2: 6/6, precision 6/7 = 86%). person, motorbike, car không ngưỡng nào ≤ 0.5 đạt 90%, nên tôi chọn theo đánh đổi: person 0.15 (recall 34/42 = 81%, precision 34/47 = 72%; ở 0.2 recall chỉ 32/42 = 76% mà precision hơn 2 điểm; lên 0.5 mất người ở xa, r02 img_007); motorbike 0.5 (recall đứng yên 6/9 từ 0.15 tới 0.5 nên lấy ngưỡng cao nhất cho phép: precision 6/10 = 60% so với 6/12 = 50% ở 0.35); car 0.5 (recall đứng yên 7/8 ở mọi ngưỡng ≤ 0.6; precision 7/13 = 54% so với 7/15 = 47% ở 0.3). bus (1 vật) và truck (0 vật) quá ít để chọn ngưỡng. |
| three_wheeler                                    | YOLOE-26s-seg, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` → three_wheeler | 0.15                                                                                                        | YOLO26 không có lớp này: tuk-tuk ở img_002 không có box, xích lô ở img_007 bị gọi motorbike, ở img_032 bị gọi bicycle (r01). YOLOE tìm đúng tuk-tuk img_002 (0.93) (r03). Ở 0.15 recall 7/7 = 100%, precision 7/9 = 78%; lên 0.25 recall rơi còn 5/7. Merge ưu tiên YOLOE trước YOLO26 để box three_wheeler thay box car/truck chồng lên.                                                                                                                                                                                                                                                                                                                                                  |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,bicycle=0.2,motorbike=0.5,car=0.5,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: **SAM 2.1 base** (`sam2.1_b.pt`, 162 MB). Bản mặc định khuyến nghị, cân bằng chất lượng mask và tốc độ: trên GPU RTX 3050 4 GB chạy 3.75 s/ảnh (r04, gồm cả YOLO26), không cần quyền truy cập như SAM 3. SAM 3 (~3.5 GB) nặng với GPU 4 GB.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  - **Sai lớp xe ba bánh:** cả **7/7 xe ba bánh trong gold** bị YOLO26 gọi sai lớp với box rất khớp (IoU 0.86–0.98): 4 tuk-tuk img_012 thành car 0.65–0.81, img_020 hai xe thành car 0.39 và 0.30, một xe thành truck 0.31. Ở ngưỡng car 0.5, 4 tuk-tuk img_012 vẫn được giữ và chiếm 4/6 box car thừa. Ngoài gold cũng vậy: tuk-tuk img_002 không có box ở cả 0.25 lẫn 0.5 (r01, r02), img_004/img_013 bị gọi car (r04), xích lô img_007 thành motorbike, img_032 thành bicycle. Đổi ngưỡng chỉ làm box xuất hiện hay biến mất, không đổi nhãn. Cách sửa: YOLOE cho three_wheeler, merge ưu tiên YOLOE (box YOLOE thay box car/truck chồng IoU ≥ 0.6), rồi người review kiểm mọi box car/truck gần xe ba bánh. Mask SAM đẹp cũng không sửa được nhãn sai (r04).
  - **Box xe máy bị người lái che:** recall motorbike đứng yên 6/9 vì 2 xe máy chở người ở img_036 bị máy vẽ box chỉ ôm **phần đuôi xe dưới lưng người lái** (điểm 0.66–0.75), trong khi gold ôm cả phần xe nhìn thấy gồm gương và tay lái nhô trên vai người lái (guideline 4.1.1). IoU < 0.5 nên tính là sót. Máy có một box ôm đủ xe (IoU 0.97 với gold) nhưng điểm chỉ 0.12. Chiếc thứ 3 là xe đỗ ở xa cao 21 px, điểm 0.08. Hạ ngưỡng không sửa được kiểu lỗi này mà chỉ thêm box thừa (ở 0.1: 24 box thừa).
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: img_012 tuk-tuk bị gọi **car 0.81** (và 3 tuk-tuk khác 0.65–0.71); img_036 box xe máy **0.75** chỉ ôm nửa dưới xe; img_002 xe bán tải bị gọi **truck 0.73** dù guideline xếp bán tải vào `car`.
  - Điểm thấp mà đúng: img_036 box ôm đủ xe máy chở người, điểm **0.12** (IoU 0.97 với gold); img_007 người lái và xe máy ở xa 0.40–0.49, bị loại khi lên 0.5 (r02); img_032 xích lô bên trái YOLOE 0.48 (r03).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:
  1. Ngưỡng được **chọn và chấm trên cùng 4 ảnh gold**, nên precision/recall máy trên gold ở ngưỡng đã chọn là con số lạc quan, không phải ước lượng độc lập.
  2. **Mẫu nhỏ**: bicycle 6, three_wheeler 7, car 8, motorbike 9, person 42 vật gold; với các lớp dưới 10 vật, một vật đúng/sai làm recall đổi 11–17 điểm %. bus (1) và truck (0) không đủ để chọn ngưỡng.
  3. person, motorbike, car **không đạt** recall 90% ở bất kỳ ngưỡng nào ≤ 0.5 (tối đa 86%, 78%, 88%): máy sót hoặc vẽ box sai kiểu mà hạ ngưỡng không cứu được, nên bước "Bỏ sót" và "Lỏng" khi review là bắt buộc.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số  | Tỉ lệ | Mẫu số                  |
| ---------- | --- | ----- | ----------------------- |
| accepted   | 462 | 81.9% | 564 box máy             |
| edited     | 26  | 4.6%  | 564 box máy             |
| deleted    | 67  | 11.9% | 564 box máy             |
| added      | 75  | 13.3% | 563 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md`, 24 ảnh batch, ngưỡng accepted IoU ≥ 0.9. Script đã **loại 23 đối tượng quanh 12 bẫy** khỏi bảng này: lỗi bẫy là lỗi nhân tạo do `inject_traps.py` cài, nếu tính vào sẽ làm máy trông tệ hơn thực tế, nên profile chỉ phản ánh lỗi thật của YOLO26m + YOLOE. Trong deleted + added có **12 trường hợp đổi lớp** (box đúng chỗ, sai lớp; script tính là 1 xoá + 1 thêm), nên số vật máy **sót thật** ≈ 75 − 12 = **63**.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - **three_wheeler** tệ nhất về độ chính xác: trong 42 box máy chỉ 7 accepted (17%), 10 edited, **25 deleted (60%)**, thêm 5 added. Ngưỡng 0.15 chọn trên 7 xe gold cho recall 100% nhưng trên batch YOLOE bắn nhiều box vào vật không phải xe ba bánh (như đã thấy ở pilot: img_002 box 0.37 vào xe bán tải, 0.40 vào xe đẩy hàng; img_032 gộp 2 xích lô).
  - **motorbike** tệ nhất về bỏ sót: chỉ 1 box bị xoá nhưng **47 xe phải vẽ thêm**, tập trung ở ảnh đông: img_018 (máy 15 → sau duyệt 27), img_035 (5 → 9), img_005 và img_029 (3 → 7). Ngưỡng 0.5 (Part 3: recall motorbike đứng yên 6/9 trên gold) loại phần lớn xe máy nhỏ, bị che trong đám đông.
  - **person** bị xoá 34/371 (9%) do ngưỡng 0.15 thấp, nhưng ít sót (12 added). **car** tốt nhất: 34/34 box máy được giữ, chỉ 5 added.
- So với gold: precision / recall của **máy** **72.6%** (61/84 box máy, KTC 95%: 62–81%) / **83.6%** (61/73, KTC 95%: 73–90%) (mẫu số: **73 đối tượng gold** vẽ tay trên 4 ảnh). Con số lạc quan vì ngưỡng chọn trên chính 4 ảnh này. 12 vật gold máy bỏ sót và 23 box thừa cho thấy accepted 82% **không** có nghĩa máy đúng 82%: accepted chỉ đếm box máy đã vẽ và được giữ, không đếm 63 vật máy không vẽ.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **5/12 = 41.7%** (KTC 95%: 19–68%; ngưỡng đạt 80% = 10/12) → **không đạt**. Theo loại: extra **3/3**, missing 1/3, loose 1/3, wrong_class **0/3**. Loại nào lọt (7 bẫy): **wrong_class** ở img_008, img_014, img_027; **missing** ở img_006, img_024; **loose** ở img_003, img_018.
  - Vì sao (bạn đã bỏ bước nào của checklist)?
    - **Bước 3 (Sai nhãn) bị bỏ với box nhỏ:** script tính 0/3 bẫy wrong_class được sửa. Thực tế ở img_008 tôi **đã đổi** box bẫy từ `three_wheeler` sang `car` nhưng thu hẹp box từ 151 px còn 77 px (phần còn lại của xe bị tuk-tuk che); box mới chỉ chồng khoảng một nửa box gốc (IoU ≈ 0.49) nên script chấm theo box gốc và tính là lọt. Hai bẫy img_014 và img_027 (người đứng xa gán `bicycle`) thì lọt thật; QC thấy img_014 #160 một người đứng xa mang nhãn `bicycle`. Tôi đọc nhãn các box lớn ở gần nhưng không đọc nhãn box nhỏ ở xa.
    - **Bước 0–1 (quét trước, Bỏ sót) chưa đủ:** chỉ 1/3 bẫy missing; img_006 thiếu một người bị che sau người khác, img_024 là ảnh dày. Đi theo box có sẵn (neo nhãn) nên vật bị che không được thấy.
    - **Bước 4 (Lỏng):** 1/3; img_003 box người cắt mất đầu và nón vẫn được giữ.
    - **Bước 2 (Thừa) làm tốt:** 3/3 box "ma" được xoá, ví dụ box truck trên bầu trời img_035.
  - Tốc độ: 50 phút / 24 ảnh = **2.08 phút/ảnh** (`time_log.csv`) cho 576 box máy (~5.2 giây/box), dưới mức ~3 phút/ảnh GUIDE khuyên. Phần lớn thời gian dành cho vẽ thêm 75 vật (47 xe máy), nên thời gian còn lại cho việc đọc nhãn và kiểm cạnh từng box máy càng ít.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Accepted **81.9%** nhưng bắt bẫy **41.7%** là dấu hiệu **thiên kiến tự động hoá**: phần lớn box tôi giữ nguyên không phải vì đã kiểm, mà vì nhãn máy trông hợp lý. Tôi bắt tốt lỗi **dễ thấy** (box trên trời, box không có vật: 3/3) nhưng bỏ qua lỗi **cần đọc kỹ** (nhãn lớp của box nhỏ, cạnh box, vật bị che). Vì vậy accepted đo mức tôi tin máy, không đo chất lượng máy; phải đọc cùng recall máy trên gold (83.6%, mục 2) và 63 vật sót thật.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay **5.0 phút/ảnh** (20 phút / 4 ảnh gold, 73 vật) vs duyệt **2.08 phút/ảnh** (50 phút / 24 ảnh, 563 vật). Theo đối tượng: ~16.4 giây/vật khi vẽ tay vs ~5.3 giây/vật khi duyệt, nhanh hơn khoảng 3 lần. Thiên lệch (bạn gán gold trước nên đã quen cảnh):
  1. Gán gold trước nên khi duyệt batch tôi đã quen CVAT và kiểu ảnh, làm batch nhanh hơn không chỉ nhờ máy.
  2. 4 ảnh gold trung bình 18 vật/ảnh, batch 23 vật/ảnh và có 6 ảnh rất dày; so theo vật công bằng hơn so theo ảnh.
  3. Duyệt nhanh một phần do duyệt chưa kỹ (mục 3: bắt bẫy 41.7%), nên 2.08 phút/ảnh là thời gian của một lần duyệt **chưa đạt chuẩn**. Duyệt đúng checklist (~3 phút/ảnh) vẫn nhanh hơn vẽ tay (5 phút/ảnh).
- Kết luận: **có, cho person và car; có điều kiện cho motorbike; chưa đáng tin cho three_wheeler**, dựa trên:
  - car 34/34 và person 328/371 box máy được giữ; với 2 lớp này sửa box máy nhanh hơn nhiều so với vẽ mới.
  - motorbike: box máy đúng (83/90 giữ) nhưng **sót 47 xe** ở ảnh đông với ngưỡng 0.5; cần ngưỡng thấp hơn cho ảnh đông hoặc dùng `ignore_region` cho cụm xa.
  - three_wheeler: 25/42 box máy bị xoá (60%) và 5 xe phải vẽ thêm; người duyệt mất công xoá box sai gần như vẽ mới. Cần prompt tốt hơn hoặc ngưỡng chọn trên mẫu lớn hơn 7 xe.
  - Recall máy trên gold 83.6% (61/73): máy sót khoảng 1/6 vật, nên bước Bỏ sót là bắt buộc.
  - Điều kiện chung: máy chỉ đáng dùng **nếu người duyệt theo đúng checklist**; lần duyệt này bắt 5/12 bẫy nên lô chưa đủ chất lượng.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Đọc nhãn mọi box, kể cả box nhỏ ở xa.** Mỗi ảnh, sau khi kiểm box lớn, bật danh sách object ở panel phải và đọc nhãn từng dòng một lượt. Kiểm tra: bẫy `wrong_class` ≥ 3/3; lỗi `sai_lop` trong QC = 0.
  2. **Quét và đếm trước khi hiện nhãn máy, chia ảnh dày thành 4 góc.** Ghi số vật đếm được theo lớp, so với số box sau duyệt; lệch ≥ 2 thì duyệt lại. Kiểm tra: bẫy `missing` ≥ 3/3; số `bo_sot` trong QC = 0.
  3. **Bước Lỏng bắt buộc phóng to và giới hạn tốc độ.** Phóng to ≥ 200% kiểm 4 cạnh mọi box > 50 px; không duyệt nhanh hơn 3 phút/ảnh (ảnh dày > 30 vật: 5 phút/ảnh), nghỉ 5 phút sau mỗi 8 ảnh, ghi vào `time_log.csv`. Kiểm tra: bẫy `loose` ≥ 2/3; tổng thời gian batch ≥ 72 phút / 24 ảnh, có dòng `break`.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh                                  | SAM (phiên bản) | Số click                                                           | Mask gồm người lái / hàng hoá?                                                         | Lỗi quan sát                                                                                                                                                                                                                         | File trong `masks/`                   |
| ------------------------------------ | --------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| img_013 (tuk-tuk có người lái)       | SAM 2.1 base    | **5** (3 dương: thân, mui, đuôi · 2 âm: thân và chân người lái)    | Người lái: **không** (đã loại). Hàng hoá: không có. Đệm ghế khách gắn cố định: có.     | 1 click chỉ ra huy hiệu (0.4% ảnh). 3 click dính người lái và người ở quán nhìn qua khung (45.4%). 6 click (thêm âm lên nền) loại người lái nhưng mất gần hết thân xe (11.5%). Bản 5 click: viền bánh trước và chắn bùn còn hơi lẹm. | `tuktuk_img013_sam21_5click.jpg`      |
| img_013 (cùng vật, so sánh)          | SAM 1 MobileSAM | 5 (cùng click)                                                     | Người lái: không                                                                       | Mask nhỏ hơn (32.3% vs 38.6% ảnh): bánh sau răng cưa, thiếu sàn bước chân, cabin trước lốm đốm.                                                                                                                                      | `tuktuk_img013_sam1mobile_5click.jpg` |
| img_032 (xích lô đỗ, không người)    | SAM 2.1 base    | **4** (3 dương: đệm ghế, khung trước, mui · 1 âm: người đứng cạnh) | Không có người lái. Đệm hoa và chăn trên ghế: **có** (nằm trong mask).                 | Mask tốt nhất trong 3 vật: đủ mui, ghế, 2 bánh trước, biển số.                                                                                                                                                                       | `xichlo_img032_sam21_4click.jpg`      |
| img_002 (tuk-tuk xanh giữa phố đông) | SAM 2.1 base    | **4** (2 dương: thân, mui · 2 âm: người lái, hành khách)           | Người lái: **không** (đã loại). Hành khách: **có**, mask vẫn dính hành khách ngồi sau. | Cabin hở nên mask lấy cả đám đông phía sau nhìn qua cabin. Thêm 2 click âm (6 click) còn tệ hơn: người lái bị dính lại vào mask.                                                                                                     | `tuktuk_img002_sam21_4click.jpg`      |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?
  - **Giúp:** 4.2.4 "hai vật chạm nhau, vùng chạm thuộc vật ở phía trước": người lái ngồi trước khung tuk-tuk là `person`, nên đặt click âm lên người lái (img_013, img_002). Bảng lớp mục 2: `three_wheeler` gồm "cả xe lẫn khung mái / thùng hàng gắn liền", nên mui phải nằm trong mask. Mục 5 (ca xe đạp chở hàng): giỏ gắn cố định thuộc xe, nên giữ đệm ghế xích lô trong mask.
  - **Thiếu:** (1) 4.2.2 dùng ngưỡng lỗ "~5% diện tích vật" nhưng không nói vùng nhìn xuyên qua cabin hở của tuk-tuk (img_002) có phải bỏ khỏi polygon không; SAM tự lấp vùng này. (2) Chưa có quy tắc cho **hành khách ngồi trong** xe ba bánh có mui (câu hỏi mở 4). (3) Chăn/đệm rời trên ghế xích lô (img_032) là hàng hoá hay thuộc xe: câu hỏi mở 3 chưa chốt.
  - **Quy tắc / lỗi mask rút ra:** (a) Click dương rải đều các phần của xe (thân, mui, đuôi), vì một click chỉ ra một bộ phận (huy hiệu img_013). (b) Click âm chỉ đặt lên **người**, tối đa 2 click; thêm click âm lên nền làm SAM bỏ luôn thân xe (img_013 6 click) hoặc dao động khó lường (img_002 6 click). Nhiều click hơn không có nghĩa mask tốt hơn. (c) Mask đúng không sửa được nhãn sai (r04: mask đẹp nhưng nhãn vẫn car). (d) Xe có cabin hở giữa đám đông (img_002) cần sửa polygon tay sau SAM.
- _(Tuỳ chọn)_ Interactor SAM trong CVAT: CVAT của tôi có **SAM 3** dạng Interactor (nuclio). Lần đầu gọi lỗi `Network is unreachable` vì sau khi Docker khởi động lại, container SAM 3 đổi cổng mà nuclio vẫn nhớ cổng cũ; redeploy function thì chạy được. Tôi đã dùng SAM 3 Interactor trên các vật này và **kết quả tương tự** bản offline SAM 2.1: mask đầu tiên bám được xe ba bánh nhưng **chưa loại hết người lái**; chỉ cần thêm **1 click** là loại hoàn toàn người lái, trong khi offline SAM 2.1 cần đúng bộ 2 click âm đặt chọn lọc. SAM 3 cũng **nhanh hơn**: mask hiện ngay sau mỗi click. Tôi chỉ so bằng mắt, không xuất mask từ CVAT để đo diện tích.

## 6. Calibration (Part 6)

- Số quyết định: **9** (`calibration_decisions.csv`: 4 `loi_may`, 4 `cau_hoi_mo`, 1 `bay_lot`); quan trọng nhất:
  - **CAL-01**: xe ba bánh, kể cả tuk-tuk có biển TAXI, luôn là `three_wheeler`. YOLO26 gọi cả 7/7 xe ba bánh trong gold là car/truck, và chữ "taxi" trong lớp `car` của v1.0 làm người cũng dễ gán nhầm.
  - **CAL-02**: box xe máy chở người phải ôm cả gương và tay lái nhô trên vai người lái (img_036); máy chỉ vẽ đuôi xe. Guideline đã đúng, đây là máy sai.
  - **CAL-07**: 3/3 bẫy sai lớp lọt là lỗi **của tôi**, không phải của guideline.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu hỏi mở 3 (đồ trong khung xe thuộc xe, CAL-04), 4 (person trong xe ba bánh: thấy đầu + vai, CAL-05), 5 (có kính thuộc mask, khoang hở bỏ khỏi polygon, CAL-06), 7 (không đặt conf chấp nhận, CAL-09); mục 2: "taxi" của `car` chỉ cho xe ≥ 4 bánh (CAL-01); mục 4.1.2: phần nhìn qua khe / nan xe tính là nhìn thấy (CAL-08); mục 5: xích lô đỗ thành hàng mỗi xe một box (CAL-03) và ví dụ xe máy chở người (CAL-02). Câu 1, 2, 6 **chưa chốt** vì chưa có ca trong 28 ảnh.
- Nguyên nhân gốc, tách theo 3 loại:
  - **Guideline mơ hồ** (cần sửa chữ): CAL-01 (chữ "taxi"), CAL-04, CAL-05, CAL-06, CAL-08, CAL-09.
  - **Máy sai, guideline đã rõ** (không đổi chữ): CAL-02 (box xe máy chỉ ôm đuôi xe), một phần CAL-03 (YOLOE sót / gộp xích lô).
  - **Tôi sai** (không đổi chữ, sửa quy trình): CAL-07 (không đọc nhãn box nhỏ ở xa). Cả 4 lỗi QC ở mục 7 thuộc loại này hoặc là bẫy tôi để lọt.

## 7. QC và quyết định (Part 6)

- Mẫu: **5** ảnh (**0** ảnh `can_review` vì lô batch không có box nào được tick + **5** ngẫu nhiên, seed 42): img_003, img_006, img_014, img_031, img_035; 114 đối tượng. Người kiểm: phiếu `qc_review.csv` do Claude điền nháp sau khi xem ảnh QC (`work/qc/review/`), tôi kiểm lại (mục 8). **Hạn chế**: người kiểm không độc lập với người duyệt, và đã biết vị trí bẫy (`traps_key.json` mở sau khi export), nên lỗi bẫy trong mẫu dễ được thấy hơn một QC thật.
- Lỗi theo **đối tượng**: **4/114 = 3.5%** · theo **ảnh**: **3/5 = 60.0%** (ngưỡng Release ≤ 2%). Lỗi bỏ sót ở img_006 là vật không có trong 114 đối tượng, nên thực chất là 4 lỗi trên 115 vật.
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại:
  1. **sai_vi_tri_kich_thuoc: 1**. img_003 #1 person: cạnh trên box ở y≈222, cắt mất đầu và nón (bẫy loose lọt).
  2. **bo_sot: 1**. img_006: thiếu 1 person ~17×32 px đứng ngay sau người đội mũ #88, bị che một phần (bẫy missing lọt).
  3. **sai_lop: 1**. img_014 #160: một người đứng xa bên phải mang nhãn `bicycle` (bẫy wrong_class lọt).
  4. **thua: 1**. img_014 #150: box person cao 19.3 px, dưới ngưỡng 20 px của guideline 4.1.6.
  - 3/4 lỗi QC là bẫy lọt, 1/4 là lỗi tôi tự thêm (box dưới 20 px). Hai ảnh sạch (img_031, img_035) có hai lỗi tôi đã sửa đúng khi duyệt: box người dắt xe đạp kéo tới đáy ảnh, và box truck giả trên trời đã bị xoá.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 3/5 ảnh lỗi cho cận trên Wilson 95% là **88%**. Mẫu 5 ảnh không đủ chứng minh ngưỡng 2%: kể cả nếu 0/5 ảnh lỗi, tỉ lệ lỗi thật vẫn có thể tới ~3/5 = 60% (quy tắc 3). Ở đây kết luận ngược lại thì chắc chắn: với 3/5 ảnh lỗi, lô **không đạt** 2% dù mẫu nhỏ.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): script kiểm được định dạng, lớp trong taxonomy, box trong ảnh, box dưới 20 px, accepted/edited/deleted/added, tỉ lệ bẫy và tự đếm phiếu QC. Script **không** kiểm được box có ôm sát vật thật không, vật mà **cả máy lẫn tôi cùng bỏ sót** ngoài 4 ảnh gold, lớp có đúng thực tế không (img_014 #160 chỉ thấy bằng mắt), và tôi có thật sự làm bước 0 của checklist không.

**Vòng 2 (sau Rework):** tôi sửa 7 ảnh (img_003, img_006, img_014, img_018, img_024, img_025, img_027: các bẫy lọt và 12 box dưới 20 px, cụm nhỏ ở img_018 gom thành `ignore_region`), mất 50 phút (`time_log.csv` dòng `rework`), xuất `labels/qc_fixed.json` (7 ảnh, 238 đối tượng) và chạy lại `build_release.py` (580 đối tượng, 0 box dưới 20 px). QC vòng 2 với mẫu mới `qc_sample.py --seed 2026 --force`: img_006, img_013, img_018, img_026, img_031, 109 đối tượng. Kết quả: lỗi theo đối tượng **0/109 = 0%**, theo ảnh **0/5 = 0%**, nhưng cận trên 95% Wilson vẫn là **43%**. Hai ảnh rất dày (img_018, img_026) chỉ kiểm được ở mức tổng quát, chưa kiểm từng cạnh box. Phiếu vòng 1 lưu ở `work/qc_review_vong1.csv`; các số vòng 1 ở trên.

**Quyết định: Rework** (giữ sau vòng 2), ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release):

1. **Hold?** Không. Guideline đủ rõ để sửa các lỗi trong mẫu: cả 4 lỗi QC vi phạm quy tắc đã có trong v1.0 (4.1.1, 4.1.6, bảng lớp), không phải do câu hỏi còn mở. Dữ liệu không hỏng: 24/24 ảnh, nhãn hợp lệ (`validate_labels`).
2. **Rework?** Có. Ảnh lỗi 3/5 = 60% > 2%; bắt bẫy 5/12 = 41.7% < 80%; lỗi lặp: 3/3 bẫy sai lớp lọt, 2/3 bẫy box lỏng lọt, 2/3 bẫy bỏ sót lọt.
3. Không đi tới Release. Sau vòng 2, mẫu 0/5 ảnh lỗi nhưng: (a) cận trên 95% là 43%, rất xa ngưỡng 2%; (b) bắt bẫy 5/12 = 41.7% < 80% nên **người duyệt chưa đáng tin**, và 19 ảnh không được sửa ở vòng 2 vẫn là kết quả của lần duyệt đó; (c) mẫu vòng 2 có 2 ảnh dày chưa kiểm được từng box. Vì vậy chưa có bằng chứng đủ để Release.

Nếu Rework / Hold, việc cần làm tiếp (bước 1 đã làm ở vòng 2):

1. ~~Sửa trong CVAT 7 bẫy lọt (img_008, img_014, img_027 sai lớp; img_003, img_018 box lỏng; img_006, img_024 bỏ sót) và các box dưới 20 px (validate_labels báo 11 box, ví dụ img_014 #150, img_018 #182–185); export các ảnh đã sửa ra `labels/qc_fixed.json`.~~ (đã làm, vòng 2)
2. Duyệt lại cả 24 ảnh cho bước Sai nhãn (đọc nhãn mọi box nhỏ ở xa) và bước Lỏng, theo quy tắc ở mục 4.
3. Áp guideline v1.1 cho nhãn đã có (CAL-01, CAL-02: kéo box xe máy chở người lên tới gương nếu còn box kiểu máy).
4. Chạy lại `build_release.py`, `qc_sample.py --force --seed` mới (mẫu khác), điền lại `qc_review.csv`, `qc_report.py`; ghi kết quả vòng 2 vào đây. Tỉ lệ bắt bẫy 41.7% **giữ nguyên** là số đo của lần duyệt này; vòng sửa sau khi biết đáp án không được tính lại thành số bắt bẫy.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Claude Code (trợ lý AI trong VS Code) để cài đặt, chạy script, tạo Task / nạp nhãn trên CVAT qua API, xem ảnh kết quả (ảnh `viz/`, mask, ảnh QC) và viết nháp `model_log.csv`, `calibration_decisions.csv`, `guideline_v1.1`, phiếu `qc_review.csv` và các mục REPORT. Tôi tự vẽ gold, tự duyệt 24 ảnh batch trong CVAT, bấm giờ và thử SAM 3 Interactor. Cách kiểm tra: đối chiếu mọi con số với file đầu ra của script (`review_stats.md`, `threshold_sweep.csv`, `qc_report_auto.md`, `time_log.csv`) và đọc lại từng mục. Ảnh của bộ dữ liệu đã được gửi tới dịch vụ AI đám mây khi Claude xem ảnh.
- **Gold có bị nhiễm không?** Gold nộp (73 box) được vẽ tay trên Task trống trước khi chạy model, không nạp file, không dùng AI Tools (CVAT ghi 73/73 box `source = manual`, không có `conf`).
- **Máy / CVAT:** Windows 11, RTX 3050 Laptop 4 GB (CUDA), CVAT v2.81.0 Docker + SAM 3 (nuclio). Vướng mắc: SAM 3 mất kết nối khi Docker đổi cổng (sửa bằng redeploy); `cvat-cli` không tạo được nút YOLO trên CVAT 2.81.
- **Dùng cache `data/cache/`?** Không, mọi model chạy trên máy.
- **Lỗi / bug script:** Không sửa script nào.
