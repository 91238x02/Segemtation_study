import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 모델 인풋에 넣기 적합한 이미지로 변환
def mk_test_image(path, size=(256,256), input_=True):
    test_img1 = cv2.imread(path)
    test_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2RGB)/255
    test_img1 = cv2.resize(test_img1, dsize=(size[0], size[1]), interpolation=cv2.INTER_AREA)
    if input_ == True:
        if len(test_img1.shape) == 3:
            test_img1 = np.expand_dims(test_img1, axis=0)
            
    return test_img1


# 21채널로 예측된 이미지를 COLORMAP과 매핑
def color_mapping(pred_img, color_map, size=(256,256)):
    argmax_img = np.argmax(pred_img, axis=-1)
    color_pred = np.zeros((size[0],size[1], 3))
    for index_label, color in enumerate(color_map):
        bool_mask = (argmax_img == index_label)
        color_pred[bool_mask] = color
        
    return color_pred/255


# X, Y 이미지 비교
def show_diff(X, Y):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1,2,1)
    plt.title('Ground Truth')
    plt.imshow(Y)
    ax = plt.subplot(1,2,2)
    plt.title('Prediction')
    plt.imshow(X)

    
# 채널별로 나눠진 클래스 마스크를 시각화.
def display_multiple_img(mask1, VOC_CLASSES, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for label, i in zip(VOC_CLASSES, range(rows*cols)):
        ax.ravel()[i].imshow(mask1[:,:,i], cmap='gray')
        ax.ravel()[i].set_title(label)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()



# 마스크를 클래스에 맵핑하여 클래스갯수만큼 depth생성
def mk_seg_mask(gt_path, VOC_COLORMAP):
    gt1_cv2 = cv2.imread(gt_path)
    mask = cv2.cvtColor(gt1_cv2, cv2.COLOR_BGR2RGB)
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        
    return segmentation_mask


# 학습용 이미지, 마스크 생성(categorical_crossentropy)
def img_mask_set(img_path, mask_path, VOC_COLORMAP, IMG_SIZE=(256,256)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_LINEAR)/255
    img = np.array(img, dtype=np.float32)
    
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (IMG_SIZE[0], IMG_SIZE[1]), cv2.INTER_LINEAR)
    
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.uint8)
    for label_index, label in enumerate(VOC_COLORMAP):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(np.uint8)
    mask = np.array(mask/255, dtype=np.float32)
    
    return img, segmentation_mask, mask


# 학습 데이터셋(이미지, 마스크)를 구성
def dataset(img_dataset_abspath, mask_dataset_abspath, start_cnt, BATCH_SIZE, VOC_CLASSES, VOC_COLORMAP, IMG_SIZE=(256,256)):

    batch_img = np.empty((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3))
    batch_seg_mask = np.empty((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], len(VOC_CLASSES)))
    batch_mask = np.empty((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3))

    batch_img_path = img_dataset_abspath[start_cnt:BATCH_SIZE]
    batch_img_path = sorted(glob.glob(os.path.join(img_dataset_abspath, '*')))
    batch_seg_mask_path = sorted(glob.glob(os.path.join(mask_dataset_abspath, '*')))

    for bat in range(BATCH_SIZE):
        img_b, seg_mask_b, mask_b = img_mask_set(batch_img_path[bat], batch_seg_mask_path[bat], VOC_COLORMAP, IMG_SIZE)
        batch_img[bat] = img_b
        batch_seg_mask[bat] = seg_mask_b
        batch_mask[bat] = mask_b
        
    return batch_img, batch_seg_mask, batch_mask

