bash run.sh -x 132 -y 125 -d imagenette -u 0 -b 0.1 >> output132.log 2>&1 &&
bash run.sh -x 133 -y 133 -d imagenette -u 0 -b 0.1 -p >> output133.log 2>&1 &&

bash run.sh -x 134 -y 125 -d imagenette -u 0 -b 1.0 >> output134.log 2>&1 &&
bash run.sh -x 135 -y 133 -d imagenette -u 0 -b 1.0 >> output135.log 2>&1 &&
bash runlast.sh -x 136 -y 125 -d imagenette -u 0 -c 1 -b 0.01 >> output136.log 2>&1 &&
bash runlast.sh -x 137 -y 133 -d imagenette -u 0 -c 1 -b 0.01 >> output137.log 2>&1 &&
bash runlast.sh -x 138 -y 125 -d imagenette -u 0 -c 1 -b 0.1 >> output138.log 2>&1 &&
bash runlast.sh -x 139 -y 133 -d imagenette -u 0 -c 1 -b 0.1 >> output139.log 2>&1 &&
bash runlast.sh -x 140 -y 125 -d imagenette -u 0 -c 1 -b 1.0 >> output140.log 2>&1 &&
bash run.sh -x 141 -y 133 -d imagenette -u 0 -c 1 -b 1.0 >> output141.log 2>&1


# 本地 y 125是robust model(也传到了galaxy上), 133是clean model

# below to run on HAL, try different lamda
nohup bash run.sh -x 142 -y 142 -d imagenette -u 0 -l 300 -p -r /home/haoyang/SRe2L/ >> output142.log 2>&1 &
nohup bash run.sh -x 143 -y 143 -d imagenette -u 1 -l 500 -p -r /home/haoyang/SRe2L/ >> output143.log 2>&1 &

bash run.sh -x 144 -y 143 -d imagenette -u 2 -r /home/haoyang/SRe2L/ >> output143.log 2>&1 &&


nohup bash run.sh -x 149 -y 125 -d imagenette -u 0 -c 1 -b 10.0 >> output149.log 2>&1 &&
nohup bash run.sh -x 150 -y 133 -d imagenette -u 0 -c 1 -b 10.0 >> output150.log 2>&1 &&
nohup bash run.sh -x 151 -y 133 -d imagenette -u 0 -c 1 -b 1.0 >> output151.log 2>&1


bash run.sh -x 190 -y 190 -d imagenette -u 0 -c 10 -p -n >> output190.log 2>&1 &&
bash run.sh -x 191 -y 190 -d imagenette -u 0 -c 10 >> output191.log 2>&1 &&
bash run.sh -x 192 -y 190 -d imagenette -u 0 -c 10 -w >> output192.log 2>&1 &&
bash run.sh -x 193 -y 190 -d imagenette -u 0 -c 10 -n -w >> output193.log 2>&1


bash run.sh -x 200 -y 190 -d imagenette -u 0 -c 10 -b 0.001 -w -n >> output200.log 2>&1 &&
bash run.sh -x 201 -y 190 -d imagenette -u 0 -c 10 -b 0.1 -w -n >> output201.log 2>&1 &&
bash run.sh -x 202 -y 190 -d imagenette -u 0 -c 10 -w >> output202.log 2>&1

#bash run.sh -x 194 -y 190 -d imagenette -u 0 -c 10 -n ->> output194.log 2>&1 &&
#bash run.sh -x 195 -y 190 -d imagenette -u 0 -c 10 -n >> output195.log 2>&1 &&

bash run.sh -x 184 -y 184 -d tiny-imagenet -u 0 -c 10 -p -n -g >> output184.log 2>&1 &&

nohup bash run.sh -x 190 -y 190 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -p -n >> output190.log 2>&1 &
nohup bash run.sh -x 191 -y 190 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' >> output191.log 2>&1 &
nohup bash run.sh -x 192 -y 190 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -w >> output192.log 2>&1 &
nohup bash run.sh -x 193 -y 190 -d tiny-imagenet -u 3 -c 10 -r '/home/hl57/data/' -w -n >> output193.log 2>&1 &
# 以上是server上debug用的

nohup bash run.sh -x 195 -y 195 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -p -n >> output195.log 2>&1 &
nohup bash run.sh -x 196 -y 195 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' >> output196.log 2>&1 &
nohup bash run.sh -x 197 -y 195 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -w >> output197.log 2>&1 &
nohup bash run.sh -x 198 -y 195 -d tiny-imagenet -u 3 -c 10 -r '/home/hl57/data/' -w -n >> output198.log 2>&1 &
# 以上是server上的2x2实验，已完成


bash run.sh -x 205 -y 190 -d imagenette -u 0 -c 10 -n >> stdout/205.log 2>&1 &&
bash run.sh -x 206 -y 190 -d imagenette -u 0 -c 10 >> stdout/206.log 2>&1 &&
bash run.sh -x 207 -y 190 -d imagenette -u 0 -c 10 -w >> stdout/207.log 2>&1 &&
bash run.sh -x 208 -y 190 -d imagenette -u 0 -c 10 -n -w >> stdout/208.log 2>&1

bash run.sh -x 1001 -y 1 -d imagenette -u 0 -c 10 -n -g > stdout/1001.log &&
bash run.sh -x 1002 -y 1 -d imagenette -u 0 -c 10 -g > stdout/1002.log &&
bash run.sh -x 1003 -y 1 -d imagenette -u 0 -c 10 -n -w -g > stdout/1003.log


bash run.sh -x 206 -y 206 -d imagenette -u 0 -c 10 -p -n > stdout/206.log
bash run.sh -x 207 -y 206 -d imagenette -u 0 -c 10 > stdout/207.log 2>&1 &&
bash run.sh -x 208 -y 206 -d imagenette -u 0 -c 10 -w > stdout/208.log 2>&1 &&
bash run.sh -x 209 -y 206 -d imagenette -u 0 -c 10 -n -w > stdout/209.log 2>&1

bash run.sh -x 214 -y 214 -d imagenette -u 0 -c 10 -p -n > stdout/214.log
bash run.sh -x 215 -y 214 -d imagenette -u 0 -c 10 > stdout/215.log 2>&1 &&
bash run.sh -x 216 -y 214 -d imagenette -u 0 -c 10 -w > stdout/216.log 2>&1 &&
bash run.sh -x 217 -y 214 -d imagenette -u 0 -c 10 -n -w > stdout/217.log 2>&1

bash run.sh -x 218 -y 214 -d imagenette -u 0 -c 10 -b 0.001 -n -w > stdout/218.log 2>&1 &&
bash run.sh -x 219 -y 214 -d imagenette -u 0 -c 10 -b 0.1 -n -w > stdout/219.log 2>&1

bash run.sh -x 220 -y 1 -d tiny-imagenet -u 0 -c 1 -p >> stdout/220.log 2>&1 &&
bash run.sh -x 221 -y 1 -d tiny-imagenet -u 0 -c 10 -p >> stdout/221.log 2>&1


nohup bash run.sh -x 210 -y 195 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -n >> output210.log 2>&1 &
nohup bash run.sh -x 211 -y 195 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' >> output211.log 2>&1 &
nohup bash run.sh -x 212 -y 195 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -w >> output212.log 2>&1 &
nohup bash run.sh -x 213 -y 195 -d tiny-imagenet -u 3 -c 10 -r '/home/hl57/data/' -w -n >> output213.log 2>&1 &


bash run2.sh -x 221 -y 1 -d tiny-imagenet -u 0 -c 10 >> stdout/221.log 2>&1


bash run.sh -x 226 -y 226 -d tiny-imagenet -u 0 -c 10 -p > stdout/226.log 2>&1 &&
bash run.sh -x 227 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w > stdout/227.log 2>&1 &&
bash run.sh -x 228 -y 226 -d tiny-imagenet -u 0 -c 10 -n > stdout/228.log 2>&1 &&
bash run.sh -x 229 -y 226 -d tiny-imagenet -u 0 -c 10 -w > stdout/229.log 2>&1

bash run.sh -x 230 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w > stdout/230.log 2>&1


bash run.sh -x 235 -y 214 -d imagenette -u 0 -c 10 -n -w > stdout/235.log 2>&1

bash run.sh -x 235 -y 214 -d imagenette -u 0 -c 10 -n -w -a 0.01 -b 10 > stdout/235.log 2>&1


bash run.sh -x 236 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -a 0.01 -b 10 > stdout/236.log 2>&1 &&
bash run.sh -x 237 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 10 > stdout/237.log 2>&1 &&
bash run.sh -x 238 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 0.1 > stdout/238.log 2>&1 &&
bash run.sh -x 239 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -a 0.01 > stdout/239.log 2>&1 &&
bash run.sh -x 240 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -a 1.0 > stdout/240.log 2>&1


bash run.sh -x 240 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -a 1.0 > stdout/240.log 2>&1

bash run.sh -x 241 -y 214 -d imagenette -u 0 -c 10 -n -w > stdout/241.log 2>&1

bash run.sh -x 244 -y 214 -d imagenette -u 0 -c 10 -n -w > stdout/244.log 2>&1

bash run.sh -x 251 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/251.log 2>&1



bash run.sh -x 257 -y 226 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 3 > stdout/257.log 2>&1 &
bash run.sh -x 258 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 10 > stdout/258.log 2>&1 &
bash run.sh -x 259 -y 226 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 30 > stdout/259.log 2>&1 &


bash run.sh -x 264 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' > stdout/264.log 2>&1 &


bash runlast.sh -x 251 -y 214 -d imagenette -e resnet50 -u 0 -c 10 -n -w -b 0.1 > stdout/268.log 2>&1

bash run.sh -x 269 -y 255 -d imagenet -u 0 -c 10 -n -w -b 0.1 > stdout/269.log 2>&1 &

bash run.sh -x 269 -y 255 -d imagenet -u 0 -c 10 -n -w -b 0.1


bash run.sh -x 275 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/275.log 2>&1 &&
bash run.sh -x 276 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.3 > stdout/276.log 2>&1 &&
bash run.sh -x 277 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 > stdout/277.log 2>&1
# 这是为了看weighted barycenter的效果, 为了省时间，把计算barycenter的n_iter设成了20左右，效果不好

bash run.sh -x 278 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/278.log 2>&1
bash run.sh -x 279 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/279.log 2>&1
# 278: weighted， n_iter设成100看看效果 ; 279 unweighted。weighted还是略差

bash run.sh -x 281 -y 255 -d imagenet -u 0 -c 10 -n -w -b 0.3
# 这是为了分析get_bary的运行时间，找galaxy上过于缓慢的原因

bash run.sh -x 282 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 -g > stdout/282.log 2>&1 &&
bash run.sh -x 283 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/283.log 2>&1 &&
bash run.sh -x 283 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/284.log 2>&1
# 以上是为了试刚写的自动存取weight和barycenter的功能

bash run.sh -x 296 -y 226 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' > stdout/296.log 2>&1 &
bash run.sh -x 292 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 3 > stdout/292.log 2>&1 &
bash run.sh -x 293 -y 226 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 10 > stdout/293.log 2>&1 &
bash run.sh -x 294 -y 226 -d tiny-imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 30 > stdout/294.log 2>&1 &
# 10/22 以上是正在galaxy上跑的


nohup bash run.sh -x 304 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 0.1 > stdout/304.log 2>&1 & # marked
nohup bash run.sh -x 305 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 1.0 > stdout/305.log 2>&1 & # marked
nohup bash run.sh -x 306 -y 226 -d tiny-imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.03 > stdout/306.log 2>&1 &
nohup bash run.sh -x 307 -y 226 -d tiny-imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 50 > stdout/307.log 2>&1 &


bash run.sh -x 308 -y 214 -d imagenette -u 0 -c 1 > stdout/308.log 2>&1 &&
bash run.sh -x 309 -y 214 -d imagenette -u 0 -c 1 -n -w -b 1.0 > stdout/309.log 2>&1 &&
bash run.sh -x 310 -y 214 -d imagenette -u 0 -c 1 -n -w -b 3.0 > stdout/310.log 2>&1 &&
bash run.sh -x 311 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 > stdout/311.log 2>&1 &&
bash run.sh -x 312 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 > stdout/312.log 2>&1 &&
bash run.sh -x 313 -y 226 -d tiny-imagenet -u 0 -c 1 > stdout/313.log 2>&1 &&
bash run.sh -x 314 -y 226 -d tiny-imagenet -u 0 -c 1 -n -w -b 10 > stdout/314.log 2>&1 &&
bash run.sh -x 315 -y 226 -d tiny-imagenet -u 0 -c 1 -n -w -b 30 > stdout/315.log 2>&1
# 实验 1 ipc的情况，并与更大的r_bn接轨。

bash run.sh -x 326 -y 214 -d imagenette -u 0 -c 1 -n -w -b 3.0 > stdout/326.log 2>&1 &&
bash run.sh -x 327 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 > stdout/327.log 2>&1 &&
bash run.sh -x 328 -y 226 -d tiny-imagenet -u 0 -c 1 -n -w -b 30 > stdout/328.log 2>&1
# 把first-bn-multiplier从10减到1，看看效果

bash run.sh -x 331 -y 214 -d imagenette -u 0 -c 1 -n -w -b 3.0 > stdout/331.log 2>&1 &&
bash run.sh -x 332 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 > stdout/332.log 2>&1 &&
bash run.sh -x 333 -y 226 -d tiny-imagenet -u 0 -c 1 -n -w -b 30 > stdout/333.log 2>&1
# first-bn-multiplier还是10，只要第一层，去掉其他层的bn，看看效果

bash run.sh -x 334 -y 214 -d imagenette -u 0 -c 1 -n -w -b 3.0 > stdout/334.log 2>&1 &&
# first-bn-multiplier是0，不要第一层，保留其他层的bn，看看效果

nohup bash run.sh -x 337 -y 226 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 30 > stdout/337.log 2>&1 &
# 看来把first-bn-multiplier从10减到1最好。看看10 ipc的效果


nohup bash run.sh -x 320 -y 226 -d tiny-imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.3 > stdout/320.log 2>&1 &
nohup bash run.sh -x 321 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 10 -a 0.3 > stdout/321.log 2>&1 &
nohup bash run.sh -x 322 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 3.0 > stdout/322.log 2>&1 &
nohup bash run.sh -x 323 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 10.0 > stdout/323.log 2>&1 &
# galaxy上继续调参，争取tiny-imagenet上的improvement大一点，imagenet上做出improvement


nohup bash run.sh -x 329 -y 214 -d imagenette -u 0 -c 50 -r '/home/hl57/data/' > stdout/329.log 2>&1 &
nohup bash run.sh -x 330 -y 226 -d tiny-imagenet -u 1 -c 50 -r '/home/hl57/data/' > stdout/330.log 2>&1 &


nohup bash run.sh -x 339 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 10.0 > stdout/339.log 2>&1 & # marked
nohup bash run.sh -x 340 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 30.0 > stdout/340.log 2>&1 &
# first_bn_multiplier改成1, 继续跑


bash run.sh -x 341 -y 214 -d imagenette -u 0 -c 50 -n -w -b 3.0 > stdout/341.log 2>&1


# ---- 以下是10月24日的

nohup bash run.sh -x 343 -y 255 -d imagenet -u 0 -c 1 -r '/home/hl57/data/' > stdout/343.log 2>&1 &
nohup bash run.sh -x 344 -y 255 -d imagenet -u 2 -c 1 -r '/home/hl57/data/' -n -w -b 10 > stdout/344.log 2>&1 &
nohup bash run.sh -x 348 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 50 > stdout/348.log 2>&1 &

# ImageNet-1k开启1 ipc实验；10ipc继续调参

nohup bash run.sh -x 349 -y 255 -d imagenet -u 2 -c 1 -r '/home/hl57/data/' -n -w -b 30 > stdout/349.log 2>&1 &
nohup bash run.sh -x 350 -y 255 -d imagenet -u 0 -c 1 -r '/home/hl57/data/' -n -w -b 50 > stdout/350.log 2>&1 &


# Tiny-ImageNet调参
nohup bash run.sh -x 352 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 50 > stdout/352.log 2>&1 &


# ImageNet-1k 1 ipc调参

nohup bash run.sh -x 353 -y 255 -d imagenet -u 0 -c 1 -r '/home/hl57/data/' -n -w -b 30 -a 0.1 > stdout/353.log 2>&1 &
nohup bash run.sh -x 355 -y 255 -d imagenet -u 2 -c 1 -r '/home/hl57/data/' -n -w -b 3 -a 0.1 > stdout/355.log 2>&1 &


# 试了squared L-2，很多run之后发现性能没以前好，放弃

# bash run.sh -x 367 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.01 > stdout/367.log 2>&1 &&
bash run.sh -x 372 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10.0 -a 0.1 > stdout/372.log 2>&1 &&
bash run.sh -x 373 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10.0 -a 0.03 > stdout/373.log 2>&1 &&
bash run.sh -x 374 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10.0 -a 0.01 > stdout/374.log 2>&1

# 下面保守一些，继续调参
nohup bash run.sh -x 361 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.5 > stdout/361.log 2>&1 &
nohup bash run.sh -x 362 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 1.0 > stdout/362.log 2>&1 &
nohup bash run.sh -x 363 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 50 -a 0.5 > stdout/363.log 2>&1 &
nohup bash run.sh -x 364 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 50 -a 1.0 > stdout/364.log 2>&1 &
# 还是不保守了
nohup bash run.sh -x 368 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.03 > stdout/368.log 2>&1 &
nohup bash run.sh -x 369 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.1 > stdout/369.log 2>&1 &
nohup bash run.sh -x 370 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.3 > stdout/370.log 2>&1 &
nohup bash run.sh -x 371 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 10 -a 0.1 > stdout/371.log 2>&1 &

#
nohup bash run.sh -x 375 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 100 -a 0.1 > stdout/375.log 2>&1 &
nohup bash run.sh -x 376 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 100 -a 0.3 > stdout/376.log 2>&1 &
nohup bash run.sh -x 377 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 300 -a 0.1 > stdout/377.log 2>&1 &  # 这个最好
nohup bash run.sh -x 378 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 300 -a 0.3 > stdout/378.log 2>&1 &

nohup bash run.sh -x 401 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 1000 -a 0.3 > stdout/401.log 2>&1 &
nohup bash run.sh -x 402 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 1000 -a 0.1 > stdout/402.log 2>&1 &
nohup bash run.sh -x 403 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 1000 -a 0.03 > stdout/403.log 2>&1 &
nohup bash run.sh -x 404 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 1000 -a 0.01 > stdout/404.log 2>&1 &



# bash run.sh -x 367 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.01 > stdout/367.log 2>&1 &&
bash run.sh -x 379 -y 214 -d imagenette -u 0 -c 10 -n -w -b 30.0 -a 0.1 > stdout/379.log 2>&1 &&
bash run.sh -x 380 -y 214 -d imagenette -u 0 -c 10 -n -w -b 30.0 -a 0.03 > stdout/380.log 2>&1 &&
bash run.sh -x 381 -y 214 -d imagenette -u 0 -c 10 -n -w -b 30.0 -a 0.01 > stdout/381.log 2>&1 &&

bash run.sh -x 382 -y 214 -d imagenette -u 0 -c 10 -n -w -b 30.0 -a 0.3 > stdout/382.log 2>&1 &&
bash run.sh -x 383 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10.0 -a 0.3 > stdout/383.log 2>&1

bash run.sh -x 384 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.3 > stdout/384.log 2>&1 &&
bash run.sh -x 385 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.1 > stdout/385.log 2>&1 &&
bash run.sh -x 386 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.03 > stdout/386.log 2>&1 &&
bash run.sh -x 387 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.01 > stdout/387.log 2>&1


# 10/29/2023
bash run.sh -x 416 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.3 > stdout/416.log 2>&1 &&
bash run.sh -x 417 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.1 > stdout/417.log 2>&1 &&
bash run.sh -x 418 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.03 > stdout/418.log 2>&1 &&
bash run.sh -x 419 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3.0 -a 0.01 > stdout/419.log 2>&1

bash run.sh -x 420 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 -a 0.3 > stdout/420.log 2>&1 &&
bash run.sh -x 421 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 -a 0.1 > stdout/421.log 2>&1 &&
bash run.sh -x 422 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 -a 0.03 > stdout/422.log 2>&1 &&
bash run.sh -x 423 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1.0 -a 0.01 > stdout/423.log 2>&1

bash run.sh -x 424 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.3 -a 0.3 > stdout/424.log 2>&1 &&
bash run.sh -x 425 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.3 -a 0.1 > stdout/425.log 2>&1 &&
bash run.sh -x 426 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.3 -a 0.03 > stdout/426.log 2>&1 &&
bash run.sh -x 427 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.3 -a 0.01 > stdout/427.log 2>&1


nohup bash run.sh -x 411 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 50 -a 0.3 > stdout/411.log 2>&1 &  (wd: ~/SRe2L)
nohup bash run.sh -x 413 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 0.3 > stdout/413.log 2>&1 &  (wd: ~/SRe2L)
nohup bash run.sh -x 415 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 30 -a 1.0 > stdout/415.log 2>&1 &  (wd: ~/SRe2L)


bash run.sh -x 444 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3 > stdout/444.log 2>&1 &&
bash run.sh -x 445 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10 > stdout/445.log 2>&1

bash run.sh -x 454 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3 > stdout/454.log 2>&1 &&
bash run.sh -x 455 -y 214 -d imagenette -u 0 -c 10 -n -w -b 10 > stdout/455.log 2>&1


nohup bash run.sh -x 457 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 50 > stdout/457.log 2>&1 &
nohup bash run.sh -x 458 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 50 -a 0.1 > stdout/458.log 2>&1 &
nohup bash run.sh -x 459 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 50 -a 0.5 > stdout/459.log 2>&1 &
nohup bash run.sh -x 460 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 100 > stdout/460.log 2>&1 &  # marked

nohup bash run.sh -x 461 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 100 -a 0.1 > stdout/461.log 2>&1 &
nohup bash run.sh -x 462 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 100 -a 0.5 > stdout/462.log 2>&1 &
nohup bash run.sh -x 463 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -w -b 200 > stdout/463.log 2>&1 &
nohup bash run.sh -x 464 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 300 > stdout/464.log 2>&1 &

nohup bash run.sh -x 465 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 500 > stdout/465.log 2>&1 & # (marked)
nohup bash run.sh -x 466 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -b 1000 > stdout/466.log 2>&1 & # (marked)

nohup bash run.sh -x 468 -y 255 -d imagenet -u 2 -c 50 -r '/home/hl57/data/' > stdout/468.log 2>&1 &
nohup bash run.sh -x 470 -y 226 -d tiny-imagenet -u 3 -c 100 -r '/home/hl57/data/' > stdout/470.log 2>&1 &

# 以上在galaxy上跑imagenet实验


bash run.sh -x 470 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 30> stdout/470.log 2>&1 &&
bash run.sh -x 471 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 30 -a 0.25 > stdout/471.log 2>&1 &&
bash run.sh -x 472 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 30 -a 0.03 > stdout/472.log 2>&1 &&
bash run.sh -x 473 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 50 > stdout/473.log 2>&1 &&
# bash run.sh -x 474 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 50 -a 0.25 > stdout/474.log 2>&1 &&
# bash run.sh -x 475 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 50 -a 0.03 > stdout/475.log 2>&1
# 以上在本地跑tiny-imagenet实验

bash run.sh -x 476 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 100 > stdout/476.log 2>&1 &&
bash run.sh -x 477 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 300 > stdout/477.log 2>&1 &&  # Ours_v2_10ipc
bash run.sh -x 478 -y 226 -d tiny-imagenet -u 0 -c 10 -n -w -b 500 > stdout/478.log 2>&1

bash run.sh -x 479 -y 214 -d imagenette -u 0 -c 1 -n -w -b 30 > stdout/479.log 2>&1 &&
bash run.sh -x 480 -y 214 -d imagenette -u 0 -c 1 -n -w -b 100 > stdout/480.log 2>&1 &&
bash run.sh -x 481 -y 214 -d imagenette -u 0 -c 1 -n -w -b 300 > stdout/481.log 2>&1 &&

bash run.sh -x 482 -y 214 -d imagenette -u 0 -c 1 -n -w -b 500 > stdout/482.log 2>&1 &&
bash run.sh -x 483 -y 214 -d imagenette -u 0 -c 1 -n -w -b 1000 > stdout/483.log 2>&1 &&
bash run.sh -x 484 -y 214 -d imagenette -u 0 -c 10 -n -w -b 500 > stdout/484.log 2>&1 &&
bash run.sh -x 485 -y 214 -d imagenette -u 0 -c 10 -n -w -b 1000 > stdout/485.log 2>&1

# conflict: server side 478 and 480 are about more IPC experiment. Rename local 478 with 478_local


nohup bash run.sh -x 486 -y 255 -d imagenet -u 0 -c 1 -r '/home/hl57/data/' -n -w -b 500 > stdout/486.log 2>&1 &
nohup bash run.sh -x 487 -y 255 -d imagenet -u 1 -c 1 -r '/home/hl57/data/' -n -w -b 50 > stdout/487.log 2>&1 &
nohup bash run.sh -x 488 -y 255 -d imagenet -u 2 -c 1 -r '/home/hl57/data/' -n -w -b 10 > stdout/488.log 2>&1 &
nohup bash run.sh -x 489 -y 226 -d tiny-imagenet -u 3 -c 50 -r '/home/hl57/data/' -n -w -b 300 > stdout/489.log 2>&1 &

bash run.sh -x 490 -y 214 -d imagenette -u 0 -c 100 > stdout/490.log 2>&1 &&
bash run.sh -x 491 -y 214 -d imagenette -u 0 -c 100 -n -w -b 500 > stdout/491.log 2>&1 &&
bash run.sh -x 492 -y 214 -d imagenette -u 0 -c 100 -n -w -b 10 > stdout/492.log 2>&1

# Local experiment, id 493, 494cross-architecture generalization on ImageNette
bash runlast.sh -x 484 -y 214 -e resnet50 -d imagenette -u 0 -c 10 > stdout/493.log 2>&1 &&
bash runlast.sh -x 484 -y 214 -e resnet101 -d imagenette -u 0 -c 10 > stdout/494.log 2>&1

# cross-architecture and ablation
nohup bash runlast.sh -x 465 -y 255 -e resnet50 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' > stdout/495.log 2>&1 &
nohup bash runlast.sh -x 465 -y 255 -e resnet101 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' > stdout/496.log 2>&1 &
nohup bash run.sh -x 497 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n > stdout/497.log 2>&1 &
nohup bash run.sh -x 498 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -w > stdout/498.log 2>&1 &



bash run.sh -x 499 -y 214 -d imagenette -u 0 -c 1 -n -w -b 500 > stdout/499.log 2>&1 &&
bash run.sh -x 500 -y 214 -d imagenette -u 0 -c 10 -n -w -b 500 > stdout/500.log 2>&1 &&
bash run.sh -x 501 -y 214 -d imagenette -u 0 -c 10 -n > stdout/501.log 2>&1 &&
bash run.sh -x 502 -y 214 -d imagenette -u 0 -c 10 -w > stdout/502.log 2>&1 &&
bash run.sh -x 503 -y 214 -d imagenette -u 0 -c 10 -n -b 500 > stdout/503.log 2>&1 &&
bash run.sh -x 504 -y 214 -d imagenette -u 0 -c 10 -w -b 500 > stdout/504.log 2>&1 &&
bash run.sh -x 505 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.001 > stdout/505.log 2>&1 &&
bash run.sh -x 506 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.01 > stdout/506.log 2>&1 &&
bash run.sh -x 507 -y 214 -d imagenette -u 0 -c 10 -n -w -b 0.1 > stdout/507.log 2>&1 &&
bash run.sh -x 508 -y 214 -d imagenette -u 0 -c 10 -n -w -b 2000 > stdout/508.log 2>&1 &&
bash run.sh -x 509 -y 214 -d imagenette -u 0 -c 10 -n -w -b 3000 > stdout/509.log 2>&1


#nohup bash run.sh -x 510 -y 214 -d imagenette -u 0 -c 10 -r '/home/hl57/data/' > stdout/510 .log 2>&1 &
nohup bash run.sh -x 511 -y 214 -d imagenette -u 0 -c 10 -r '/home/hl57/data/' > stdout/511 2>&1 &
nohup bash run.sh -x 512 -y 214 -d imagenette -u 1 -c 50 -r '/home/hl57/data/' > stdout/512 2>&1 &

nohup bash run.sh -x 513 -y 214 -d imagenette -u 0 -c 50 -r '/home/hl57/data/' -n -w -b 500 > stdout/513 2>&1 &

#haoyang@dt-login02.delta.ncsa.illinois.edu:~
# try with zca, but failed
bash run.sh -x %d -y 214 -d imagenette -u 0 -c 10 -n -w -b 500 > stdout/%d.log 2>&1
nohup bash run.sh -x %d -y 214 -d imagenette -u 0 -c 10 -r '/home/hl57/data/' -n -w -b 500 > stdout/%d.log 2>&1 &

# 这两行跑错了，本想结合的，却忘了w
bash run.sh -x 514 -y 214 -d imagenette -u 0 -c 10 -n -t 1.0 > stdout/514.log 2>&1 &&
bash run.sh -x 515 -y 214 -d imagenette -u 0 -c 10 -n -t 0.1 > stdout/515.log 2>&1 &&

# 把r_wb用来调整barycenter loss的强度，loss_ce强度为1
bash run.sh -x 517 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.01 > stdout/517.log 2>&1 &&
bash run.sh -x 518 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.03 > stdout/518.log 2>&1 &&
bash run.sh -x 519 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.1 > stdout/519.log 2>&1 &&
bash run.sh -x 520 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.3 > stdout/520.log 2>&1 &&
bash run.sh -x 521 -y 214 -d imagenette -u 0 -c 10 -n -w -t 1.0 > stdout/521.log 2>&1 &&
bash run.sh -x 522 -y 214 -d imagenette -u 0 -c 10 -n -w -t 3.0 > stdout/522.log 2>&1

bash run.sh -x 533 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.003 > stdout/533.log 2>&1 &&
bash run.sh -x 534 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.001 > stdout/534.log 2>&1

# 把r_wb用来调整loss_ce的强度,
bash run.sh -x 535 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0.1 -b 500 > stdout/535.log 2>&1 &&
bash run.sh -x 536 -y 214 -d imagenette -u 0 -c 10 -n -w -t 1 -b 500 > stdout/536.log 2>&1 &&
bash run.sh -x 537 -y 214 -d imagenette -u 0 -c 10 -n -w -t 3 -b 500 > stdout/537.log 2>&1 &&
bash run.sh -x 538 -y 214 -d imagenette -u 0 -c 10 -n -w -t 10 -b 500 > stdout/538.log 2>&1 &&
bash run.sh -x 539 -y 214 -d imagenette -u 0 -c 10 -n -w -t 30 -b 500 > stdout/539.log 2>&1 &&
bash run.sh -x 540 -y 214 -d imagenette -u 0 -c 10 -n -w -t 100 -b 500 > stdout/540.log 2>&1


nohup bash run.sh -x 529 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -t 0.0 -b 500 > stdout/529.log 2>&1 &  # 38.24
nohup bash run.sh -x 530 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -t 0.1 -b 500 > stdout/530.log 2>&1 &  # 37.45
# nohup bash run.sh -x 531 -y 255 -d imagenet -u 2 -c 50 -r '/home/hl57/data/' -n -w -t 1.0 -b 500 > stdout/531.log 2>&1 &

nohup bash run.sh -x 541 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -n -w -t 0.01 -b 500 > stdout/541.log 2>&1 &  # 38.27
nohup bash run.sh -x 542 -y 255 -d imagenet -u 1 -c 10 -r '/home/hl57/data/' -n -w -t 1.0 -b 500 > stdout/542.log 2>&1 &  # 38.04

nohup bash run.sh -x 544 -y 255 -d imagenet -u 2 -c 10 -r '/home/hl57/data/' -n -b 500 > stdout/544.log 2>&1 &  # 37.36

nohup bash run.sh -x 532 -y 255 -d imagenet -u 3 -c 50 -r '/home/hl57/data/' -n -b 500 > stdout/532.log 2>&1 &  #

nohup bash run.sh -x 999 -y 255 -d imagenet -u 0 -c 10 -r '/u/haoyang/data/' -n -b 500 > stdout/999.log 2>&1 &

bash slurm.sh -x 545 -y 226 -d tiny-imagenet -c 100 -r '/u/haoyang/data/' -n -w -t 0 -b 300
bash slurm.sh -x 546 -y 226 -d tiny-imagenet -c 10 -r '/u/haoyang/data/' -n -w -t 0.1 -b 300
bash run.sh -x 999 -y 255 -d imagenet -c 10 -r '/scratch/bcac/dataSet/' -n -w -t 1 -b 300

bash run.sh -x 999 -y 226 -d tiny-imagenet -c 10 -r '/u/haoyang/data/' -n -w -t 1 -b 300

srun -A bcac-delta-gpu --time=00:30:00 --nodes=1 --ntasks-per-node=16 \
--partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=16g --pty /bin/bash


nohup bash run.sh -x 561 -y 255 -d imagenet -u 0 -c 1 -r '/home/hl57/data/' -n -w -t 1 -b 500 >> stdout/561.log 2>&1 &
nohup bash run.sh -x 562 -y 255 -d imagenet -u 1 -c 1 -r '/home/hl57/data/' -n -w -t 10 -b 500 >> stdout/562.log 2>&1 &
nohup bash run.sh -x 563 -y 255 -d imagenet -u 2 -c 1 -r '/home/hl57/data/' -n >> stdout/563.log 2>&1 &
nohup bash run.sh -x 564 -y 255 -d imagenet -u 3 -c 1 -r '/home/hl57/data/' -n -b 500 >> stdout/564.log 2>&1 &


nohup bash run.sh -x 570 -y 255 -d imagenet -u 2 -c 50 -r '/home/hl57/data/' -n -w -t 0.0 -b 500 > stdout/570.log 2>&1 &
nohup bash run.sh -x 571 -y 255 -d imagenet -u 3 -c 100 -r '/home/hl57/data/' -n -w -t 0.0 -b 500 > stdout/571.log 2>&1 &


bash run.sh -x 572 -y 214 -d imagenette -u 0 -c 1 -n -w -t 1 -b 10 > stdout/572.log 2>&1 &&
bash run.sh -x 573 -y 214 -d imagenette -u 0 -c 1 -n -w -t 0 -b 10 > stdout/573.log 2>&1 &&
bash run.sh -x 574 -y 214 -d imagenette -u 0 -c 1 -n -b 10 > stdout/574.log 2>&1 &&
bash run.sh -x 575 -y 214 -d imagenette -u 0 -c 1 -n > stdout/575.log 2>&1 &&
bash run.sh -x 576 -y 214 -d imagenette -u 0 -c 10 -n -w -t 1 -b 10 > stdout/576.log 2>&1 &&
bash run.sh -x 577 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0 -b 10 > stdout/577.log 2>&1 &&
bash run.sh -x 578 -y 214 -d imagenette -u 0 -c 10 -n -b 10 > stdout/578.log 2>&1 &&
bash run.sh -x 579 -y 214 -d imagenette -u 0 -c 10 -n > stdout/579.log 2>&1 &&
bash run.sh -x 580 -y 214 -d imagenette -u 0 -c 50 -n -w -t 1 -b 10 > stdout/580.log 2>&1 &&
bash run.sh -x 581 -y 214 -d imagenette -u 0 -c 50 -n -w -t 0 -b 10 > stdout/581.log 2>&1 &&
bash run.sh -x 582 -y 214 -d imagenette -u 0 -c 50 -n -b 10 > stdout/582.log 2>&1 &&
bash run.sh -x 583 -y 214 -d imagenette -u 0 -c 50 -n > stdout/583.log 2>&1


# nohup bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e resnet50 -u 0 -c 50 -r '/home/hl57/data/' -n -w -b 300 > stdout/584.log 2>&1 &
nohup bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e resnet101 -u 1 -c 50 -r '/home/hl57/data/' > stdout/584.log 2>&1 &

# 100 ipc 60.74

bash run.sh -x 592 -y 255 -d imagenet -c 50 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500  # starting 902, quota exceed
bash run.sh -x 593 -y 255 -d imagenet -c 100 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500  # starting 452, quota exceed
bash run.sh -x 594 -y 255 -d imagenet -c 100 -r '/scratch/bcac/dataSet/'


bash run.sh -x 595 -y 214 -d imagenette -u 0 -c 10 -n -w -t 0 -b 10 > stdout/595.log 2>&1


# some details: actually, I use 'data_synthesis_new.py' whenever per_class_bn is true, because this setting in the older file was not well implemented.
bash run.sh -x 809 -y 809 -d imagenette -m ViT -e ViT -p -u 0 -c 10 -n -w -t 0 -b 10 > stdout/809.log 2>&1

bash runlast.sh -x 809 -y 214 -d imagenette -e ViT -u 0 -c 10 -n -w -t 0 -b 10 > stdout/810.log 2>&1


# ablation
nohup bash run.sh -x 811 -y 255 -d imagenet -u 0 -c 10 -r '/home/hl57/data/' -w -b 500 >> stdout/811.log 2>&1 &
nohup bash runlast.sh -x 468 -y 255 -d imagenet -e resnet50 -u 0 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 500 >> stdout/857.log 2>&1 &
nohup bash run.sh -x 812 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -w -b 500 > stdout/812.log 2>&1 &

nohup bash runlast.sh -x 468 -y 255 -d imagenet -e vit -u 0 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 500 >> stdout/883.log 2>&1 &
nohup bash runlast.sh -x 468 -y 255 -d imagenet -e resnet50 -u 0 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 500 >> stdout/857.log 2>&1 &

# cross-architecture
nohup bash runlast.sh -x 529 -y 255 -d imagenet -e vit -u 2 -c 10 -r '/home/hl57/data/' -n -w -t 0 -b 500 > stdout/817.log 2>&1 &
# nohup bash runlast.sh -x 529 -y 255 -d imagenet -e vits -u 1 -c 10 -r '/home/hl57/data/' -n -w -t 0 -b 500 > stdout/845.log 2>&1 & too weak, deprecated
# 853
nohup bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e vit -u 1 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 300 > stdout/853.log 2>&1 &

# 819
nohup bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e resnet50 -u 3 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 300 > stdout/819.log 2>&1 & # 54 at the 110th iteration
nohup bash runlasttwo.sh -x 843 -y 226 -d tiny-imagenet -e resnet101 -u 2 -c 50 -r '/home/hl57/data/' -n -w -t 0 -b 300 > stdout/843.log 2>&1 &
# synthetic data copied from 489


# cross-arch
# 813, 814, 818
"""
bash run.sh -x 820 -y 255 -d imagenet -c 10 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 0.1
bash runlast.sh -x 821 -y 255 -d imagenet -e resnet50 -c 50 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500
bash runlast.sh -x 822 -y 255 -d imagenet -e resnet101 -c 50 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500
bash runlast.sh -x 823 -y 255 -d imagenet -e vit -c 50 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500
"""

bash runlasttwo.sh -x 852 -y 226 -d tiny-imagenet -e resnet101 -c 50 -r '/u/haoyang/data/' -n -w -t 0 -b 500
# 854
bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e vit -c 50 -r '/u/haoyang/data/' -n -w -t 0 -b 500
# 855
bash runlast.sh -x 489 -y 226 -d tiny-imagenet -e vits -c 50 -r '/u/haoyang/data/' -n -w -t 0 -b 500

# bash runlast.sh -x 824 -y 255 -d imagenet -e vits -c 50 -r '/scratch/bcac/dataSet/' -n -w -t 0 -b 500
# cross-architecture
# nohup bash runlast.sh -x 813 -y 226 -d tiny-imagenet -e resnet50 -u 2 -c 10 -r '/home/hl57/data/' -w -b 500 > stdout/813.log 2>&1 &



bash runlast.sh -x 577 -y 214 -d imagenette -e vit -u 0 -c 10 -n -w -t 0 -b 10 > stdout/815.log 2>&1
bash runlast.sh -x 577 -y 214 -d imagenette -e vits -u 0 -c 10 -n -w -t 0 -b 10 > stdout/990.log 2>&1

bash run.sh -x 576 -y 214 -d imagenette -u 0 -c 10 -n -w -t 1 -b 10 > stdout/576.log 2>&1 &&
# local
#bash runlasttwo.sh -x 478 -y 226 -d tiny-imagenet -e resnet50 -u 0 -c 10 -n -w -t 0 -b 500 > stdout/835.log 2>&1 &&
#bash runlast.sh -x 478 -y 226 -d tiny-imagenet -e resnet101 -u 0 -c 10 -n -w -t 0 -b 500 > stdout/836.log 2>&1 &&
bash runlasttwo.sh -x 478 -y 226 -d tiny-imagenet -e vits -u 0 -c 10 -n -w -t 0 -b 500 > stdout/837.log 2>&1 &&
bash runlast.sh -x 478 -y 226 -d tiny-imagenet -e vit -u 0 -c 10 -n -w -t 0 -b 500 > stdout/838.log 2>&1

nohup bash run.sh -x 885 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -w -n -t 0.0 -b 0.1 >> stdout/885.log 2>&1 &
nohup bash run.sh -x 886 -y 226 -d tiny-imagenet -u 1 -c 10 -r '/home/hl57/data/' -w -n -t 0.0 -b 0.3 >> stdout/886.log 2>&1 &
r_bn: [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

Imagenet:
[12.27, x, 26.61, 31.98, 35.55, 35.76, 37.11, 37.68, 37.53]

Imagenette:
[, , 58.85, 61.2, 63.23, , 64.31, ]

bash run.sh -x 887 -y 214 -d imagenette -u 0 -c 10 -w -n -t 0.0 -b 0.3 >> stdout/887.log 2>&1 &&
bash run.sh -x 888 -y 214 -d imagenette -u 0 -c 10 -w -n -t 0.0 -b 30 >> stdout/888.log 2>&1 &&
bash run.sh -x 889 -y 214 -d imagenette -u 0 -c 10 -w -n -t 0.0 -b 300 >> stdout/889.log 2>&1


nohup bash run.sh -x 890 -y 255 -d imagenet -u 3 -c 10 -r '/home/hl57/data/' -n -w -b 0.1 > stdout/890.log 2>&1 &

bash run1.sh -x 902 -y 226 -d tiny-imagenet -u 0 -c 1 -b 30> stdout/902.log 2>&1


bash slurm.sh -x 2 -y 1 -d tiny-imagenet -c 1 -r '/u/haoyang/data/' -n -w -t 0 -b 30
bash slurm.sh -x 3 -y 1 -d tiny-imagenet -c 1 -r '/u/haoyang/data/' -n -w -t 0 -b 300
bash slurm.sh -x 4 -y 1 -d tiny-imagenet -c 10 -r '/u/haoyang/data/' -n -w -t 0 -b 300
bash slurm.sh -x 5 -y 1 -d tiny-imagenet -c 50 -r '/u/haoyang/data/' -n -w -t 0 -b 300

nohup bash fkd.sh -x t1 -y 226 -d tiny-imagenet -c 1 -r '/media/techt/DATA/data/' -n -w -t 0 -b 300 -s 64 > stdout/t1.log 2>&1 &
nohup bash fkd.sh -x t2 -y 226 -d tiny-imagenet -c 1 -r '/media/techt/DATA/data/' -n -w -t 0 -b 300 -s 64 > stdout/t2.log 2>&1 &

nohup bash fkd.sh -x t4 -y 226 -d tiny-imagenet -c 50 -r '/media/techt/One Touch/data/' -n -w -t 0 -b 300 -s 32 > stdout/t4.log 2>&1 &

nohup bash run.sh -x zz -y zz -d imagenette -u 0 -c 10 -r '~/data/' -w -n -t 0.0 -b 300 -p -A >> stdout/zz.log 2>&1 &
nohup bash run.sh -x z10 -y 214 -d imagenette -u 0 -c 10 -r '/media/techt/DATA/data/' -w -n -t 0.0 -b 300 >> stdout/z10.log 2>&1 &
nohup bash run.sh -x z11 -y 214 -d imagenette -u 0 -c 10 -r '/media/techt/DATA/data/' -t 0.0 -b 300 >> stdout/z11.log 2>&1 &


nohup bash run.sh -x z12 -y 214 -d imagenette -u 0 -c 1 -r '/media/techt/DATA/data/' -w -n -t 0.0 -b 300 >> stdout/z12.log 2>&1 &&
nohup bash run.sh -x z13 -y 214 -d imagenette -u 0 -c 1 -r '/media/techt/DATA/data/' -t 0.0 -b 300 >> stdout/z13.log 2>&1


nohup bash run.sh -x z16 -y 214 -d imagenette -u 0 -c 1 -r '/media/techt/DATA/data/' -w -n -t 0.0 -b 300 >> stdout/z16.log 2>&1 &&
nohup bash run.sh -x z17 -y 214 -d imagenette -u 0 -c 1 -r '/media/techt/DATA/data/' -t 0.0 -b 300 >> stdout/z17.log 2>&1


bash run.sh -x v7 -y 214 -d imagenette -u 0 -c 1 >> stdout/v7.log 2>&1 &&
bash run.sh -x v8 -y 214 -d imagenette -u 0 -c 10 >> stdout/v8.log 2>&1 &&
bash run.sh -x v9 -y 214 -d imagenette -u 0 -c 50 >> stdout/v9.log 2>&1