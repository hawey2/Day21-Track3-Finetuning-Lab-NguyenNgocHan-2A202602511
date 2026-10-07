# Reflection — Lab 21

| | |
|---|---|
| Họ tên | Nguyen Ngoc Han |
| MSSV | 2A202602511 |
| Ngày | 2026-10-07 |

### 1. Điều gì làm bạn ngạc nhiên nhất?

NB1 đo supervised fraction là **37/94 token (0.3936)**, và hai assert cùng xanh: câu trả lời có trong loss,
câu hỏi bị mask. Tôi cũng không ngờ p95 của 250 mẫu chỉ là 98 token, trong khi cấu hình CPU đặt
`max_length=512`; con số này nhắc tôi rằng độ dài chuỗi nên được đo trên dữ liệu thật thay vì chọn theo
cảm tính. Kết quả đó mới chứng minh mask và dữ liệu, chưa chứng minh fine-tune tốt hơn prompt.

### 2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?

Phần tốn thời gian nhất là dựng môi trường CPU rồi kiểm tra lại artefact NB1, không phải chạy notebook:
test suite hoàn tất nhanh (116 passed, 3 skipped), còn NB1 cần tải tokenizer từ Hugging Face. Tôi đã
nghĩ phần khó nhất sẽ là huấn luyện, nhưng bước chuẩn bị và xác minh điều kiện chạy mới là phần đầu tiên
cần xử lý. Các notebook NB2–NB5 cần GPU nên chưa chạy trên máy này.

### 3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?

Tôi từng nghĩ loss giảm là dấu hiệu đủ tốt rằng quá trình fine-tune đang đi đúng hướng. Lab nhấn mạnh
rằng loss train không trả lời được câu hỏi adapter có cải thiện target hay làm hỏng năng lực tổng quát
không. Muốn kết luận phải đóng băng baseline trước train, chấm trên cùng dữ liệu và dùng các nhóm target,
regression, format, latency. Hiện tôi mới có bằng chứng mask đúng; chưa có số để kết luận model thắng.

### 4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?

Tôi dùng AI assistant để rà soát tài liệu, đối chiếu cấu hình với mã nguồn, chạy các kiểm tra có thể thực
hiện trên CPU và hoàn thiện report/reflection từ artefact đo được. Tôi không có kết quả GPU nên yêu cầu
assistant đánh dấu rõ các bảng NB2–NB5 là chưa đo, thay vì điền số tham khảo như thể là kết quả của tôi.
Điểm cần kiểm tra kỹ là mọi câu khẳng định trong report phải truy về file `results/` hoặc được ghi rõ là
giả thuyết/tham khảo; ví dụ, số baseline đã công bố trong repo không thay thế kết quả của lần chạy này.

### 5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?

Tôi sẽ xác định tác vụ, tiêu chí thành công và các năng lực không được suy giảm trước khi chọn model hay
train. Sau đó tạo một tập eval đại diện và tách biệt khỏi train, chốt cách chấm target/regression/format/
latency, rồi đo base model với prompt mạnh. Chỉ khi biết baseline và có dữ liệu sạch, tôi mới thử một
fine-tune nhỏ; trước đó tôi sẽ chứng minh loss mask và kiểm tra prompt train có khớp prompt inference.
