# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603191 - Nguyễn Bảo Khánh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 3 | Hàng nhô ra / đồ trên xe: box và mask xe có gồm không? | Đồ **nằm gọn trong khung xe** (đệm, chăn, rổ gắn, thùng gắn) **thuộc xe**: box và mask gồm. Hàng **nhô ra ngoài khung > 20% chiều dài xe** vẫn không tính (giữ v1.0 mục 4.1.5). | img_032 (đệm hoa và chăn trên ghế xích lô) | CAL-04 |
| 4 | Người ngồi trong xe ba bánh có mui (chỉ thấy thấp thoáng): ngưỡng "≥ 20%" có hợp lý? | Thay ước lượng "20% diện tích" bằng tiêu chí nhìn thấy được: **gán `person` khi thấy rõ đầu và vai**. Box chỉ ôm phần nhìn thấy. Người lái / hành khách **không** nằm trong mask xe (vùng người thuộc `person`, mục 4.2.4). | img_013 (người lái tuk-tuk), img_002 (người lái và hành khách tuk-tuk xanh) | CAL-05 |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe không? | **Có kính** (kính chắn gió, cửa kính) thì thuộc mask xe dù nhìn xuyên thấy người/nội thất. **Khoang hở không có kính** (thành bên tuk-tuk, giữa mui và thân xe) là **lỗ lớn**: bỏ khỏi polygon theo 4.2.2, kể cả khi SAM tự lấp. | img_002 (SAM lấp cả đám đông nhìn qua cabin hở) | CAL-06 |
| 7 | Mức tin cậy tối thiểu để chấp nhận box máy không cần sửa? | **Không đặt.** `conf` không phải xác suất đúng: img_012 tuk-tuk bị gọi `car` 0.81, img_002 xe bán tải bị gọi `truck` 0.73. Mọi box đều qua checklist 4 bước; `conf` chỉ dùng để sắp thứ tự xem (box điểm thấp xem trước). | img_012, img_002 | CAL-09 |

**Chưa chốt (giữ để v1.2):**
- **Câu 1, xe đạp điện / xe máy điện nhỏ:** trong 28 ảnh đã gán (4 gold + 24 batch) tôi không gặp ca nào phân biệt được rõ là xe điện, nên không có ảnh ví dụ để chốt.
- **Câu 2, khối xe máy đỗ dày gần camera:** lô này tôi dùng `ignore_region` cho 15 cụm (ví dụ img_024: 6 vùng) theo quy tắc vật < 20 px của v1.0; chưa có đủ ca cụm **gần** camera để đặt ngưỡng cụ thể.
- **Câu 6, box chồng nhau trong ảnh đêm:** chưa đánh giá riêng ảnh đêm trong lô này.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_012, img_020, img_004 | Tuk-tuk, kể cả tuk-tuk có biển **TAXI** trên mui | YOLO26m gọi cả 7/7 xe ba bánh trong gold là `car` (img_012: 0.65–0.81) hoặc `truck`; bảng lớp v1.0 ghi `car` gồm "taxi" nên dễ hiểu nhầm | `three_wheeler`: **số bánh** quyết định lớp, không phải công dụng | **Mục 2:** dòng `car`, chữ "taxi" chỉ áp cho ô tô ≥ 4 bánh; dòng `three_wheeler` thêm "kể cả tuk-tuk có biển TAXI" (CAL-01) |
| img_007, img_032 | Xích lô đỗ sát nhau thành hàng | YOLOE-26s sót cả 3 xích lô (img_007, 0 box) và gộp 2 xích lô thành 1 box (img_032, 0.27) | Mỗi xe một box riêng, kể cả đỗ sát nhau; xe che một phần vẫn gán nếu thấy ≥ 20% | **Mục 5** (bảng ca khó): thêm dòng "xích lô đỗ thành hàng" (CAL-03) |
| img_031 | Người dắt xe đạp chở hàng, chân nhìn thấy qua nan bánh xe | YOLO26m cho box `person` dừng ở hông | Phần nhìn thấy qua khe hở / nan bánh xe vẫn là **phần nhìn thấy**: box kéo tới điểm thấp nhất nhìn thấy | **Mục 4.1.2:** thêm "nhìn qua khe, nan xe, lưới vẫn tính là nhìn thấy" (CAL-08) |
| img_036 | Xe máy chở người nhìn từ phía sau, gương và tay lái nhô trên vai người lái | YOLO26m vẽ box chỉ ôm đuôi xe dưới lưng người lái (0.66–0.75); box ôm đủ xe chỉ có điểm 0.12 | Box `motorbike` ôm mọi phần xe nhìn thấy, gồm gương và tay lái | **Không đổi:** v1.0 mục 4.1.1 đã đủ rõ; đây là máy sai. Ghi vào mục 5 làm ví dụ để người duyệt kéo box lên (CAL-02) |
| img_008, img_014, img_027 | Box sai lớp, ví dụ người đứng xa bị gán `bicycle` (bẫy wrong_class) | Bẫy cài sẵn; người duyệt để lọt 3/3 | Đổi về đúng lớp | **Không đổi:** bảng lớp v1.0 đã rõ. Đây là lỗi người duyệt; sửa bằng quy trình (đọc nhãn mọi box, kể cả box nhỏ ở xa) (CAL-07) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 3 (đồ trong khung xe thuộc xe), 4 (person trong xe ba bánh: thấy đầu + vai; không thuộc mask xe), 5 (có kính thuộc mask, khoang hở bỏ khỏi polygon), 7 (không đặt conf chấp nhận). Mục 2: tuk-tuk có biển TAXI là `three_wheeler`, "taxi" của `car` chỉ cho xe ≥ 4 bánh. Mục 4.1.2: phần nhìn qua khe / nan xe tính là nhìn thấy. Mục 5: thêm ca xích lô đỗ thành hàng và ca xe máy chở người. Câu 1, 2, 6 chưa chốt. | CAL-01, CAL-03, CAL-04, CAL-05, CAL-06, CAL-08, CAL-09. CAL-02 và CAL-07 ghi nhận nhưng **không** đổi chữ guideline (máy sai / người duyệt sai). |
