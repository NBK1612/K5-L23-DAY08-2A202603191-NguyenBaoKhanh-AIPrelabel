# Thông tin học viên

- Họ và tên: Nguyễn Bảo Khánh
- MSHV: 2A202603191
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: NBK1612
- Repo bài nộp (URL): https://github.com/NBK1612/K5-L23-DAY08-2A202603191-NguyenBaoKhanh-AIPrelabel
- Commit chốt bài (hash): de8d935b9b2f9118e2db8c1441f9d85b50383798
- Máy: Windows 11 Home, GPU NVIDIA GeForce RTX 3050 Laptop 4 GB (CUDA)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 462/564 = 81.9% / 26/564 = 4.6% / 67/564 = 11.9% (mẫu số: 564 box máy); added: 75/563 = 13.3% (mẫu số: 563 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 61/84 = 72.6% / 61/73 = 83.6% (73 đối tượng gold vẽ tay)
- Bắt bẫy: 5/12 = 41.7%
- Thời gian: gán tay 5.0 phút/ảnh (20 phút / 4 ảnh); duyệt có pre-label 2.08 phút/ảnh (50 phút / 24 ảnh)
- Lỗi QC: vòng 1: 60.0% theo ảnh (3/5), 3.5% theo đối tượng (4/114); vòng 2 sau Rework: 0% theo ảnh (0/5), 0% theo đối tượng (0/109), cận trên 95% = 43%
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 3/5 = 60% > 2% và bắt bẫy 5/12 < 80%, lỗi lặp là sai lớp box nhỏ và bỏ sót; vòng 2 sạch trên 5 ảnh nhưng cận trên 43% và người duyệt chưa đạt bẫy nên giữ Rework.
