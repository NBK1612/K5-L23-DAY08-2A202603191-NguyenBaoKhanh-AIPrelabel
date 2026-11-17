# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 564; đối tượng sau duyệt: 563; ngưỡng accepted: IoU >= 0.9
- Đã **loại 23 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 462 | 462/564 = 81.9% | 564 box máy |
| edited (sửa) | 26 | 26/564 = 4.6% | 564 box máy |
| deleted (xoá) | 67 | 67/564 = 11.9% | 564 box máy |
| added (vẽ thêm) | 75 | 75/563 = 13.3% | 563 đối tượng sau duyệt |

Trong số deleted + added, **12 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 63.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 328 | 9 | 34 | 12 |
| bicycle | 6 | 1 | 3 | 1 |
| motorbike | 83 | 6 | 1 | 47 |
| car | 34 | 0 | 0 | 5 |
| bus | 2 | 0 | 1 | 2 |
| truck | 2 | 0 | 3 | 3 |
| three_wheeler | 7 | 10 | 25 | 5 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 73 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 61 | 23 | 12 | 72.6% (KTC 95%: 62–81%) | 83.6% (KTC 95%: 73–90%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 5/12 = 41.7% (KTC 95%: 19–68%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - missing: 1/3
  - wrong_class: 0/3
  - loose: 1/3
  - lọt: img_006.jpg (missing), img_014.jpg (wrong_class), img_024.jpg (missing), img_027.jpg (wrong_class), img_018.jpg (loose), img_003.jpg (loose), img_008.jpg (wrong_class)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
