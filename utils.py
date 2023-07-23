def gammaadjust(img, out_range=[-1, 1], gamma=1):
    low_in, high_in = np.min(img), np.max(img)
    low_out, high_out = out_range[0], out_range[1]
    img_out = (((img - low_in) / (high_in - low_in)) ** gamma) * (high_out - low_out) + low_out
    return img_out

def SDA_Augment(image):
    xxx=random.random()
    if xxx<0.5:
        sigma = np.random.uniform(0.3, 0.9)
        image=gammaadjust(image,gamma=sigma)
    if xxx>=0.5:
        sigma = np.random.uniform(1.1, 2.8)
        image=gammaadjust(image,gamma=sigma)
    if random.random()<0.6:
        blur=np.random.choice(np.array([7,9,11,13,15,17]))
        image=cv2.GaussianBlur(image,ksize=(blur,blur),sigmaX=0.)
    return np.squeeze(image)

def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true_f = paddle.flatten(y_true,stop_axis=-2)[:,0:5]
    y_pred_f = paddle.flatten(y_pred,stop_axis=-2)[:,0:5]
    intersect = paddle.sum(y_true_f * y_pred_f, axis=0)
    denom = paddle.sum((y_true_f + y_pred_f), axis=0)
    return 1-paddle.mean((2. * intersect + smooth) / (denom + smooth))
