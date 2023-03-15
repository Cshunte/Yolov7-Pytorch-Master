import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
from final_ball_point import bb_intersection_over_union

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(opt):
    ###改动过的代码，加上了save_point
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, save_point = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate, opt.save_point
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    ##导入视频
    for root, dir, files in os.walk("pzp\\source"):
        for file in files:
            source = os.path.join(root, file)

    # 创建位置保存列表
    # list = deque(maxlen=1000)
    # list0=[0]
    # list.append(list0)

    dict = {}
    dict_point = {}



    # 读取第一帧(改）
    video0 = cv2.VideoCapture(source)
    ret, image0 = video0.read()
    cv2.imwrite('background_whole.jpg', image0)
    h, w, n = image0.shape
    image00 = image0[750:h, 0:w]
    cv2.imwrite('back.jpg', image00)
    img00 = image0[625:h, 0:w]
    cv2.imwrite('back_o.jpg', img00)
    # 第一帧搜索的区域
    area = [0, 0, w, h]

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        ###重新创建文件夹
        if yolo_weights == "person.pt":
            p1 = "inference/result/result_per_img"
            if not evaluate:
                if os.path.exists(p1):
                    pass
                    shutil.rmtree(p1)  # delete output folder
                os.makedirs(p1)

        if yolo_weights == "pzp\\weights\\ball.pt":
            p3 = "inference/result/per_img_ball"
            if not evaluate:
                if os.path.exists(p3):
                    pass
                    shutil.rmtree(p3)  # delete output folder
                os.makedirs(p3)

        if yolo_weights == "person.pt":
            p2 = "inference/result/per_img_player"
            if not evaluate:
                if os.path.exists(p2):
                    pass
                    shutil.rmtree(p2)  # delete output folder
                os.makedirs(p2)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    # txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    txt_path_o = str(Path(out)) + '/' + 'result_data' + '.txt'

    # 改个路径以便可以第二次检测续写
    txt_path = "inference/result/result_data.txt"
    ball_point_path = "inference/result/ball.txt"

    if yolo_weights == "person.pt":
        f0 = open(txt_path, 'w')
        f0.close()
    if yolo_weights == "pzp\\weights\\ball.pt":
        f0 = open(ball_point_path, 'w')
        f0.close()

    nu = 0
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # ball_path1=
        # ball_
        # b_name = "img_" + str(frame_idx) + ".jpg"
        # ball_path = os.path.join(ball_dir, b_name)
        # # print(ball_path)
        # cv2.imwrite(ball_path, img)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        ###改动的地方（关乎使用颜色挑选目标）
        # pred = py_cpu_softnms(pred,opt.conf_thres,opt.iou_thres,sigma=0.5,thresh=0.001,method=2)
        # print(type(pred))
        # print(pred)
        # print("______")
        # list0=[209, 164, 254]
        # print(img)
        # print(im0s)
        # print("_______")
        # for pre in pred:
        #     for pe in pre:
        #         pe0=map(int,pe[:4])
        #         print(pe0)
        #         # x1, y1, x2, y2 = DeepSort._xywh_to_xyxy(pe0)
        #         x, y, w, h = pe0
        #         x1 = max(int(x - w / 2), 0)
        #         x2 = min(int(x + w / 2), self.width - 1)
        #         y1 = max(int(y - h / 2), 0)
        #         y2 = min(int(y + h / 2), self.height - 1)
        #         print(x1)
        #
        #
        #         print(pe[:4])
        #         for p in pe:
        #             print(p)
        # pred=color_select(im0s,pred[:4],list0)
        # ########

        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            ###########################

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    ###

                    ####
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                # print(xywhs)
                # print(xywhs)
                # for i in xywhs:
                #     print(i)

                confs = det[:, 4]
                # print(confs)
                clss = det[:, 5]
                # print(clss)
                nu = nu + 1
                #########挑选颜色
                # list0 = color_load()
                # list0 = [255, 255, 255]
                # xywhs,confs,clss=color_select(nu,im0,list0,xywhs,confs,clss)

                #############改动
                ##去掉教练和黑色摄像机
                # xywhs, confs, clss = del_notwant_box(im0, xywhs, confs, clss)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                # 判断球是否在临近区域内（目的是为了减少误检）
                if len(outputs) > 0:
                    for ot in outputs:
                        bboxes = ot[0:4]
                        iou = bb_intersection_over_union(area, bboxes)
                        if iou < 0:
                            # 新的output
                            print("此处误检")
                            outputs = []
                            break
                        else:
                            #
                            outputs = []
                            outputs.append(ot)
                            # 更换搜索区域
                            w_area = abs(bboxes[2] - bboxes[0])
                            h_area = abs(bboxes[3] - bboxes[1])
                            area = [bboxes[0] - 3 * w_area, bboxes[1] - 3 * h_area, bboxes[2] + 2 * w_area,
                                    bboxes[3] + 3 * h_area]
                            break

                # #当检测不到时，跨大区域
                # w_halfarea = int(abs(area[2]-area[0])/2)
                # h_halfarea = int(abs(area[3] - area[1])/2)
                # expandarea = [area[0]-w_halfarea,area[1]-h_halfarea,area[2]+w_halfarea,area[3]+h_halfarea]
                # iou = bb_intersection_over_union(expandarea,bboxes)
                # if iou > 0:
                #     # 新的output
                #     outputs = []
                #     outputs.append(ot)
                #     # 更换搜索区域
                #     w_area = abs(bboxes[2] - bboxes[0])
                #     h_area = abs(bboxes[3] - bboxes[1])
                #     area = [bboxes[0] - 5 * w_area, bboxes[1] - 5 * h_area, bboxes[2] + 5 * w_area,
                #             bboxes[3] + 5 * h_area]

                # draw boxes for visualization

                if len(outputs) > 0:

                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        print(id)
                        # print(bboxes)

                        c = int(cls)  # integer class

                        label = f'{id} {names[c]} {conf:.2f}'
                        ###使得球的颜色为黑色
                        if names[c] == '1':
                            color = (0, 0, 0)
                            the_class = 1
                            id = 0
                        else:
                            color = compute_color_for_id(id)
                            the_class = 0
                        # print(color)

                        # id所对应的列表（改）
                        # if len(list)<id+1:
                        #     print("###############")
                        #
                        #
                        #     list_son = deque(maxlen=1000)
                        #     list_son.append(color)
                        #
                        #     list.append(list_son)
                        # i = 0
                        # for id0 in list(dict.keys()):
                        #     if id0==id:
                        #         i=1
                        #         break
                        # if i==0:
                        #     dict.setdefault(id,[]).append(color)
                        #     dict_point.setdefault(id, []).append(color)

                        ##改动
                        # plot_one_box_ch(bboxes, nu, M ,frame, im0, dict,dict_point, id, label=label, color=color, line_thickness=2)
                        plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
                        # print(im0.shape)
                        # cv2.imshow(img0,"img")
                        # cv2.waitKey(0)

                        ###########
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path_o, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1,
                                                               -1))  # label format
                        ##

                        ########为了保存球员和球的坐标，改动此处
                        if save_point:
                            per_img_path = "inference/result/result_per_img/img_" + str(frame_idx) + ".txt"
                            only_player_path = "inference/result/per_img_player/img_" + str(frame_idx) + ".txt"
                            only_ball_path = "inference/result/per_img_ball/img_" + str(frame_idx) + ".txt"
                            c1, c2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))
                            # bottom_center = int((c1[0] + c2[0]) / 2),int(c2[1])
                            center = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                            with open(txt_path, 'a') as f_p:
                                ###0代表是人,1代表球
                                f_p.write(('%g ' * 7) % (
                                frame_idx, the_class, id, bboxes[0], bboxes[1], bboxes[2], bboxes[3]))
                                # f_p.write(str(bottom_center) + ' ')
                                # f_p.write(str(bboxes) + ' ')
                                f_p.write(str(color) + '\n')
                            with open(per_img_path, 'a') as f:
                                ###每一帧保存为一个txt文件
                                f.write(('%g ' * 7) % (
                                frame_idx, the_class, id, bboxes[0], bboxes[1], bboxes[2], bboxes[3]))
                                # f.write(str(bottom_center) + ' ')

                                f.write(str(color) + '\n')
                                ###保存player的坐标集
                            if yolo_weights == "person.pt":
                                with open(only_player_path, 'a') as f_p:
                                    ###每一帧保存为一个txt文件
                                    f_p.write(('%g ' * 7) % (
                                        frame_idx, the_class, id, bboxes[0], bboxes[1], bboxes[2], bboxes[3]))
                                    # f.write(str(bottom_center) + ' ')

                                    f_p.write(str(color) + '\n')

                                ########为了单独保存球的坐标，以作后续判断
                            if yolo_weights == "pzp\\weights\\ball.pt":
                                with open(ball_point_path, 'a') as f_b:
                                    ###1代表是球,但是球的id规定为0
                                    f_b.write(('%g ' * 7) % (
                                    frame_idx, the_class, id, bboxes[0], bboxes[1], bboxes[2], bboxes[3]))
                                    # f_b.write(str(center) + ' ')
                                    f_b.write(str(color) + '\n')
                                with open(only_ball_path, 'a') as f_ba:
                                    ###每一帧保存为一个txt文件
                                    f_ba.write(('%g ' * 7) % (
                                        frame_idx, the_class, id, bboxes[0], bboxes[1], bboxes[2], bboxes[3]))
                                    # f.write(str(bottom_center) + ' ')

                                    f_ba.write(str(color) + '\n')
                #
                # cv2.imshow("xxxx", frame)
                # cv2.waitKey(0)

                ####
            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)

            ####改动

           #保存图片
            # str0 = "simple_change_img\\" + str(nu) + ".jpg"
            # cv2.imwrite(str0, frame)
            # print("ture")
            ###############

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                ww, hh, nn = im0.shape
                im00 = cv2.resize(im0, (int(hh / 2), int(ww / 2)))
                cv2.imshow(p, im00)
                b_name = "img_" + str(frame_idx) + ".jpg"
                ball_path1 = os.path.join("inference\\personanball_result_video\\ball_img", b_name)
                # print(ball_path)
                cv2.imwrite(ball_path1, im00)

                # cv2.imshow("xxx",frame)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h00 = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        ##改动
                        h = h00 - 0
                        # h = h0 - 0
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(im0)

            # 保存坐标（改）

            # 画出全部轨迹（改）
            # for id in dict:
            #     for i in range(1, len(dict[id])):
            #         cv2.circle(image0, dict[id][i], 1, dict[id][0], 2)
            #     cv2.imwrite('background.jpg', image0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    ###
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--yolo_weights', type=str, default='pzp\weights\\ball.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='D:\python_subject\yolov5\Yolov5_DeepSort_Pytorch-master\\volleyball_d.mov',
                        help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 大于该概率即显示
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_false', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_false', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 数据增强
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort_ball.yaml")
    parser.add_argument('--save_point', action='store_false', help='the bottom point of the box')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
