# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Ở temperature thấp (0.0 và 0.5): Model trả lời dựa trên các token có xác xuất cao nhất nên mang tính khuôn mẫu và ổn định. Ki temperature tăng lên (1.0 và 1.5): Các từ được chọn sẽ đa dạng và sẽ có nhiều câu trả lời đa dạng và phong phú 
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Theo mình thì sẽ chọn khoảng 0.5. Vì khi trả lời sản phẩm cần phải có câu trả lời tính chính xác nhất về sản phẩm cũng như chi tiết sản phẩm, chatbot cần chọn ra các từ có xác xuất cao nhất để hợp với câu hỏi của khách hàng tuy nhiên cần phải giữ một chút phong phú về từ vựng cũng như không bị quá khuôn mẫu nên mình thấy temperature=0.5 là hợp lý.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Giả sử bảng giá 2 modle như sau: GPT-4o output: $0.010 / 1K tokens | GPT-4o-mini output: $0.0006 / 1K tokens, vậy tỉ giá chênh lệch là 0.010 / 0.0006 = 16.67. Tức nếu các tác vụ không được sử dụng model hợp lý sẽ gây nên chi phí tăng cao.
> Trường hợp sử dụng:
+ Nên dùng GPT-4o: Khi cần xử lý các tác vụ phức tạp, yêu cầu suy luận đa bước sâu, giải toán/lập trình phức tạp, trích xuất dữ liệu đòi hỏi độ chính xác tuyệt đối và rủi ro ảo  thấp.
+ Nên dùng GPT-4o-mini: Cho các tác vụ phổ thông với lưu lượng lớn như chatbot CSKH cơ bản, tóm tắt bài viết ngắn, sửa lỗi chính tả/ngữ pháp để tối ưu chi phí và độ trễ.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Theo mình quan sát, hai phản hồi có sự khác biệt:
> + Giáo viên tiểu học: Câu từ ngắn gọn, từ ngữ đơn giản gần gũi, sử dụng các hình ảnh ẩn dụ trực quan (như cuốn sổ ghi chép chung mà cả lớp cùng giữ và không ai tự ý tẩy xóa được).
> + Chuyên gia tài chính: Câu dài, học thuật, dùng các thuật ngữ chuyên sâu (như công nghệ sổ cái phân tán DLT, mật mã học, cơ chế đồng thuận PoW/PoS, tính bất biến).
> System prompt đóng vai trò như chiếc la bàn định hình ngữ cảnh (context) và giới hạn không gian từ vựng của model, giúp điều chỉnh phong cách và độ phức tạp phù hợp chính xác với đối tượng tiếp nhận.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> + Thực nghiệm với đoạn văn tiếng Việt 106 từ: đếm qua `tiktoken` (gpt-4o) được 137 tokens, còn ước lượng thô (106 / 0.75) được 141.3 tokens -> hai con số chênh nhau khoảng ~3.1%.
> + Tiếng Việt tốn nhiều token hơn tiếng Anh cùng nội dung (đoạn trên dịch sang tiếng Anh chỉ tốn 90 tokens) vì các thuật toán tokenization (như BPE) được huấn luyện chủ yếu trên kho ngữ liệu tiếng Anh. Tiếng Việt có hệ thống dấu thanh và từ ghép nên thường bị tách nhỏ thành nhiều sub-words hoặc mã hóa theo từng byte UTF-8 thay vì trọn vẹn cả từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Theo mình, Streaming quan trọng nhất trong các ứng dụng trò chuyện trực tiếp (chat UI, trợ lý ảo tương tác) hoặc các tác vụ sinh nội dung dài (viết blog, code), vì nó giảm đáng kể TTFT (Time To First Token), giúp người dùng thấy kết quả phản hồi ngay lập tức thay vì phải ngồi chờ mòn mỏi vài chục giây. Ngược lại, non-streaming lại phù hợp hơn khi chạy các tác vụ ngầm (background jobs, batch processing), trích xuất dữ liệu trả về định dạng JSON có cấu trúc (structured outputs), hoặc khi cần gọi function/tool calling mà hệ thống cần trọn vẹn toàn bộ payload trước khi xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> + Exponential backoff giúp giãn thời gian chờ tăng dần theo cấp số nhân sau mỗi lần thử, tạo khoảng nghỉ đủ lâu để server có thời gian hạ tải và phục hồi tài nguyên khi gặp sự cố.
> + Nếu hàng nghìn client cùng retry với khoảng thời gian cố định giống nhau (ví dụ đều chờ 1s), tất cả các request sẽ đồng loạt dội ngược trở lại server tại cùng một thời điểm, gây ra hiện tượng Thundering Herd (bão request) làm server tiếp tục quá tải và sập hoàn toàn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> + System prompt: `"Bạn là trợ giảng AI thân thiện, trả lời ngắn gọn, súc tích bằng tiếng Việt và ưu tiên đưa ra ví dụ code minh họa khi giải thích."`
> + Lựa chọn từ ngữ quan trọng:
>   - "Ngắn gọn, súc tích": Giúp tiết kiệm output tokens (giảm chi phí và giảm độ trễ), tránh việc model giải thích lan man những điều không cần thiết.
>   - "Bằng tiếng Việt": Cố định ngôn ngữ phản hồi ngay từ đầu, tránh tình trạng câu hỏi song ngữ bị model trả lời lệch sang tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> + Hạn chế lớn nhất: Trợ lý chỉ có bộ nhớ ngắn hạn bị cắt cứng ở 3 lượt cuối (FIFO buffer 6 messages), dẫn đến việc mất hoàn toàn ngữ cảnh quan trọng nếu người dùng trò chuyện dài, và không lưu trữ trạng thái người dùng giữa các phiên khác nhau.
> + Đề xuất cải thiện: Tích hợp kỹ thuật Tóm tắt lịch sử (Conversation Summary Buffer) hoặc lưu ngữ cảnh vào Vector Database / SQLite. Cụ thể: khi history vượt quá ngưỡng token nhất định, dùng một lượt gọi model nhỏ (như mini/flash) tóm tắt các lượt trao đổi cũ thành một đoạn văn ngắn lưu trong `system/developer message`, vừa giữ được ngữ cảnh lâu dài vừa tối ưu chi phí token đầu vào.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
