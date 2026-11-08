# INT3401_11 – Nhóm 7 | Machine Learning

Dự án triển khai Project 5 – Machine Learning bằng Python và PyTorch. Nhóm hoàn thiện các mô hình và quy trình huấn luyện cho sáu bài toán:

| Phần | Bài toán |
| --- | --- |
| Q1 | Perceptron phân loại nhị phân |
| Q2 | Hồi quy phi tuyến, xấp xỉ hàm sin |
| Q3 | Mạng neural phân loại chữ số viết tay MNIST |
| Q4 | RNN nhận diện ngôn ngữ từ một từ |
| Q5 | Phép tích chập và CNN phân loại chữ số |
| Q6 | Scaled dot-product attention với causal mask |

Repository cũng có giao diện Streamlit để chạy mô phỏng và trực quan hóa các
mô hình Q1–Q6.

**Đề bài gốc:** [CS188 Project 5 – Machine Learning](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/)

## Chạy dự án từ GitHub

### 1. Clone repository

Mở Terminal hoặc PowerShell và chạy:

```bash
git clone https://github.com/dvuong2106/INT3401_11_Nhom7.git
cd INT3401_11_Nhom7
```

### 2. Tạo môi trường và cài dependencies

Dự án được phát triển với **Python 3.12**. Các lệnh dưới đây tạo môi trường
riêng cho dự án để tránh ảnh hưởng đến các gói Python khác trên máy.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell không cho phép kích hoạt môi trường, có thể chọn interpreter
`.venv\Scripts\python.exe` trong VS Code rồi chạy các lệnh `python` bằng
interpreter đó.

### 3. Tải dữ liệu

Dữ liệu không được lưu trong repository. Khi đã kích hoạt môi trường, chạy:

```bash
python setup_data.py
```

Lệnh này tải dữ liệu MNIST và nhận diện ngôn ngữ từ Berkeley vào thư mục
`data/`. Cần có kết nối Internet trong lần tải đầu tiên.

### 4. Kiểm tra chương trình

Kiểm tra các dependencies:

```bash
python autograder.py --check-dependencies
```

Chạy toàn bộ các bài kiểm tra Q1–Q6:

```bash
python autograder.py --no-graphics
```

Có thể chỉ chạy một phần, ví dụ:

```bash
python autograder.py -q q1 --no-graphics
```

Thay `q1` bằng `q2`, `q3`, `q4`, `q5` hoặc `q6` để kiểm tra phần tương ứng.
Điểm và trạng thái kiểm thử được in trong Terminal ở mục **Provisional grades**.
Một số phần cần huấn luyện mô hình nên có thể chạy lâu hơn.

### 5. Chạy giao diện mô phỏng

Sau khi cài dependencies và tải dữ liệu, chạy:

```bash
streamlit run streamlit_app.py
```

Mở địa chỉ local mà Streamlit in ra trong Terminal. Giao diện cho phép xem các
mô phỏng Q1–Q6; một số tab sẽ huấn luyện mô hình khi người dùng yêu cầu.
Dừng ứng dụng bằng `Ctrl+C` trong Terminal.

## Kết quả tham khảo

Nhóm đã ghi nhận **25/25 điểm** khi chạy autograder cho Q1–Q6. Do một số mô hình
được huấn luyện với khởi tạo ngẫu nhiên, kết quả khi chạy lại có thể khác.

## Mã nguồn chính

- `models.py`: định nghĩa mô hình và các phép tính.
- `train.py`: các vòng lặp huấn luyện.
- `losses.py`: các hàm mất mát.
- `backend.py`, `autograder.py`: dataset, tiện ích và bộ kiểm thử của đề bài.
- `streamlit_app.py`: giao diện mô phỏng Q1–Q6.
- `setup_data.py`: tải các tệp dữ liệu cần thiết.
