## Giới thiệu

Đồ án xây dựng thuật toán seam carving tuần tự (CPU) và song song (GPU) để đối chiếu tốc độ thực thi trên hai phương pháp đó
<br>
Seam Carving (hoặc liquid rescaling) là một thuật toán để thay đổi kích thước hình ảnh nhận biết nội dung, được phát triển bởi Shai Avidan, thuộc Phòng thí nghiệm Nghiên cứu Điện tử Mitsubishi (MERL) và Ariel Shamir, thuộc Trung tâm Liên ngành và MERL. Nó hoạt động bằng cách thiết lập một số seam (đường dẫn ít quan trọng nhất) trong một hình ảnh và tự động loại bỏ seam để giảm kích thước hình ảnh hoặc chèn seam để mở rộng nó. Seam Carving cũng cho phép xác định thủ công các khu vực không thể sửa đổi pixel và có khả năng xóa toàn bộ đối tượng khỏi ảnh.

Mục đích của thuật toán là nhắm mục tiêu lại hình ảnh, đây là vấn đề hiển thị hình ảnh mà không bị biến dạng trên các phương tiện có kích thước khác nhau (điện thoại di động, màn hình chiếu) bằng cách sử dụng các tiêu chuẩn tài liệu, như HTML, đã hỗ trợ các thay đổi động trong bố cục trang và văn bản nhưng không hỗ trợ hình ảnh .
<br>
Báo cáo: https://github.com/dvnhanh/seam-carving/blob/main/Seam_Carving.ipynb

## Mô tả thuật toán seam carving

![alt](img/lake_shrink.gif)

<br>
<table class="wikitable" style="max-width: 100%; overflow-y: scroll">
<tbody><tr>
<th>Step</th>
<th>Image
</th></tr>
<tr>
<td>1) Bắt đầu với một hình ảnh.	
</td>
<td><img src="https://user-images.githubusercontent.com/108814937/233795637-72f1c925-4fff-48d8-9ff3-4135b3028132.png">
</td></tr>
<tr>
<td>2) Tính weight/density/energy của mỗi pixel.  Điều này có thể được thực hiện bằng các thuật toán khác nhau: gradient magnitude, entropy, visual saliency, eye-gaze movement. Ở đây chúng tôi sử dụng gradient magnitude.
</td>
<td><img src="https://user-images.githubusercontent.com/108814937/233795644-4e02ec02-caa2-4a04-823a-dcdff6d95575.png">
</td></tr>
<tr>
<td>3) Từ độ quan trọng, lập danh sách các seam. Các seam được sắp xếp theo độ quan trọng, với các seam có độ quan trọng thấp so với hình ảnh. Các seam có thể được tính toán thông qua phương pháp dynamic programming.
</td><td>
<img src="https://user-images.githubusercontent.com/108814937/233795651-523d4969-bcf0-41e5-ac15-b6f96a54e601.png">
</td></tr>
<tr>
<td>4) Loại bỏ các seam có độ quan trọng thấp khi cần thiết.
</td>
<td><img src="https://user-images.githubusercontent.com/108814937/233795656-ddbb6995-2729-4574-9aaa-0e669ec38507.png">
</td></tr>
<tr>
<td>5) Hình ảnh cuối cùng.
</td><td>
<img src="https://user-images.githubusercontent.com/108814937/233795660-f4a46937-7934-4474-886f-b3c3ccf13c21.png"></td></tr></tbody></table>

# Thành Viên

-   Đàm Văn Nhanh ( Trưởng nhóm )
-   Nguyễn Duy Nam

## Minh họa

<img src="https://github.com/dvnhanh/seam-carving/blob/master/img/castle.jpg" height="342"> <img src="https://github.com/dvnhanh/seam-carving/blob/master/img/castle_shrink.jpg?raw=true" height="342">
