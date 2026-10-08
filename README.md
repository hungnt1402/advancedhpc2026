USTH Advanced Programming with Python 2026
===============================================

* Your name: **Nguyễn Thanh Hùng**
* Your id: **2540056**

Lab 3–10 chạy độc lập trên Google Colab; lab 2 giữ nguyên.

1. Mở notebook của lab trên Colab.
2. Chọn **Runtime → Change runtime type → T4 GPU** rồi **Run all**.
3. Tải `outputs/LabN.zip` trong **Files**: ảnh kết quả, biểu đồ theo đề, CSV số liệu và báo cáo LaTeX. Giải nén rồi biên dịch file `Report.*.tex` để lấy PDF có số liệu thực tế.

Có thể tải `image.jpg` trước khi chạy (lab 6 thêm `image2.jpg`); nếu không có, notebook tự dùng ảnh mẫu. Ảnh thứ hai được chỉnh về cùng kích thước.

GPU được đo sau warm-up và đồng bộ, không tính truyền dữ liệu; mỗi cấu hình chạy ba lần. Cần GPU thật trên Colab để lấy số liệu tốc độ.

PDF trong repo mô tả cách làm; báo cáo xuất từ Colab có thêm số liệu và hình của lần chạy đó.

Notebook cài [Numba CUDA với bộ thư viện CUDA 12](https://nvidia.github.io/numba-cuda/user/installation.html).


Project
======================

* Goals: In this project, you're to implement Non-local means denoising algorimth on an input image using CUDA.
    * Input: RGB image
    * Output: Denoised RGB image
    * Method: **Non-local means** [(Wikipedia refrence)](https://en.wikipedia.org/wiki/Non-local_means)
    * Platform: NVIDIA GPU, CUDA library, numba.

* Requirements:
    * Make a new private repo on your Github account, invite me (my email @usth) to collaborate.
    * Put everything in a *single* .py source code file. No Jupyter notebook or Google Colab.
        * Takes input file name (an image in JPG format) from ```argv[1]```
        * Writes denoised output to ```output.jpg```
        * Static filter parameters (described in extras, below):
            * Search window size $R = 21$
            * Patch size $k = 7$
            * Smoothing strength $h = 10$
    * No *extra* library (even ```cv2```!), except ```numba``` for CUDA and ```matplotlib``` for loading/saving image. I don't want to install ANY other packages in my automated test environment.
    * Don't copy paste from your friends. I have my own similarity checking tool. *I kill friendships*.
    * Responsible AI support. I will interview personally and directly, if I suspect abuse of AI usage.

* Deadline: 23:59, Sunday, October 11th 2026.
    * Hard deadline. No extension at all.
    * I will get the latest commit which is before the above deadline.


Non-local means extras:
==========================

To denoise a pixel $i$ from the input image $I$, the algorithm calculates a weighted average of all pixels $j$ within a large search window $S$ (size $R \times R$) around pixel $i$. Denote $I(i)$ as the intensity of the pixel $i$ in the input image $I$. Let $\Phi(i)$ be the output pixel intensity that corresponding to the input pixel $I(i)$. $\Phi(i)$ can *simply* calculated as:

$\Phi(i) = \frac{\sum _{j\in S}w(i,j) \cdot I(j)}{\sum _{j\in S} w(i,j)}$

The weight $w(i,j)$ evaluates how similar the local neighborhood around pixel $i$ is to the neighborhood around pixel $j$. Each neighborhood is called a comparison patch $P$ (size $k \times k$). The weight $w(i,j)$ is calculated as an exponential weight of the distance:

$w(i,j)=\exp \left(-\frac{d^{2}(i,j)}{h^{2}}\right)$

in which, $d(i,j)$ is the L2 Norm Distance and $h$ is a smoothing strength (higher $h$ leads to more blurry image). For two patches centered at $i$ and $j$ (called $P(i)$ and $P(j)$, respectively), compute the sum of squared differences for all corresponding pixels within the patch dimension:

$d^{2}(i,j)=\sum _{k\in P}\left|I(i+k)-I(j+k)\right|{}^{2}$

To take into account color RGB images, you will have to sum the differences of each channel. If the patches are nearly identical, the distance $d$ is close to 0, making the weight $w$ close to 1.
