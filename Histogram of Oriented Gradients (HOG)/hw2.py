import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    filter_x=np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    filter_y=np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    return filter_x, filter_y


def filter_image(im, filter):
    h, w = im.shape
    im_filtered = np.zeros((h, w))
    im = np.insert(im, 0, 0, axis=1)
    im = np.insert(im, 0, 0, axis=0)
    im = np.insert(im, w, 0, axis=1)
    im = np.insert(im, h, 0, axis=0)
    height, width = im.shape
    for w in range(width-2):
        for h in range(height-2):
            x = im[h:h+3, w:w+3]
            im_filtered[h, w] = np.sum(np.multiply(filter, x))

    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(np.add(np.power(im_dx, 2),np.power(im_dy, 2)))
    grad_angle = np.arctan2(im_dy,im_dx)

    for i in range(len(grad_angle)):
        for j in range(len(grad_angle[0])):
            if grad_angle[i][j]<0:
                grad_angle[i][j] += np.pi
            if grad_angle[i][j] == np.pi:
                grad_angle[i][j] = 0

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    height, width = grad_mag.shape
    M = height//cell_size
    N = width//cell_size
    ori_histo = np.zeros((M, N, 6))
    for i in range(M*cell_size):
        for j in range(N*cell_size):
            angle = grad_angle[i][j]

            if angle<np.pi/12 or angle>=11*np.pi/12:
                c = 0
            else:
                c = int((angle - np.pi/12)//(np.pi/6) + 1)

            m = i//cell_size
            n = j//cell_size
            ori_histo[m][n][c] += grad_mag[i][j]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    M, N, C = ori_histo.shape
    M_1 = M-block_size+1
    N_1 = N-block_size+1
    ori_histo_normalized = np.zeros((M_1, N_1, 6*block_size**2))
    for m in range(M_1):
        for n in range(N_1):  
            h1 = ori_histo[m,n]/np.sqrt(np.sum(np.power(ori_histo[m,n], 2)+ 0.001 ** 2))
            h2 = ori_histo[m,n+1]/np.sqrt(np.sum(np.power(ori_histo[m,n+1], 2)+ 0.001 ** 2))
            h3 = ori_histo[m+1,n]/np.sqrt(np.sum(np.power(ori_histo[m+1,n], 2)+ 0.001 ** 2))
            h4 = ori_histo[m+1,n+1]/np.sqrt(np.sum(np.power(ori_histo[m+1,n+1], 2)+ 0.001 ** 2))
            ori_histo_normalized[m,n,:] = np.append(np.append(np.append(h1, h2), h3), h4)

    return ori_histo_normalized


def extract_hog(im):
    im = im.astype('float') / 255.0
    filter_x, filter_y = get_differential_filter()
    # print(im.shape)
    # print()

    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    # plt.imshow(im_dx, cmap="Blues_r", vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(im_dy, cmap='Blues_r', vmin=0, vmax=1)
    # plt.show()
    # print(im_dx.shape)
    # print(im_dy.shape)
    # print()

    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # cv2.imwrite("grad_mag.png", grad_mag)
    # cv2.imwrite("grad_angle.png", grad_angle)
    # plt.imshow(grad_mag, cmap='Blues_r', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(grad_angle, cmap='Blues_r', vmin=0, vmax=1)
    # plt.show()
    # print(grad_mag.shape)
    # print()
    # print(grad_angle.shape)
    # print()

    cell_size = 8
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    # print(ori_histo.shape)
    # print()

    block_size = 2
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    # print(ori_histo_normalized.shape)
    # print()
    hog = np.array(ori_histo_normalized.flatten())
    # print(hog.shape)
    # visualize to verify
    #visualize_hog(im, hog, cell_size, block_size)
    return hog


def face_recognition(I_target, I_template, extract_hog_func):
  tm_h, tm_w = I_template.shape
  tr_h, tr_w = I_target.shape

  tm_hog = extract_hog(I_template)
  tm_m = np.mean(tm_hog)

  h = tr_h - tm_h
  w = tr_w - tm_w
  print(h, w)

  bounding_boxes = np.array([[0,0,0]])

  for i in range(h):
    print(i)
    for j in range(w):
      tr_hog = extract_hog(I_target[i:i+tm_h, j:j+tm_w])
      tr_m = np.mean(tr_hog)
      pr = 0
      a = 0
      b = 0

      for ii in range(tr_hog.shape[0]):
        a += np.sum((tr_hog[ii] - tr_m) ** 2)
        b += np.sum((tm_hog[ii] - tm_m) ** 2)
        pr += np.dot(tr_hog[ii] - tr_m, tm_hog[ii] - tm_m)

      ncc = pr/np.sqrt(a * b)

      if (ncc > 0.5):
          bounding_boxes = np.append(bounding_boxes, [[j,i,ncc]], axis=0)


  bounding_boxes = np.delete(bounding_boxes, 0, 0)
  bb_3 = []
  for k in range(5):
    max_ncc = bounding_boxes[:,2]
    ind = np.argmax(max_ncc)
    if k==4:
      bb_3.append(bounding_boxes[ind])
      break
    bb_2 = []
    for i in bounding_boxes:
      if i[0] > bounding_boxes[ind][0] + tm_h/2 or i[0] < bounding_boxes[ind][0] - tm_h/2:
        bb_2.append(i)
    bb_3.append(bounding_boxes[ind])
    bounding_boxes = np.array(bb_2)

  bounding_boxes = np.array(bb_3)

  return bounding_boxes


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)
    im = im.astype('float') / 255.0
    visualize_hog(im, hog, 8, 2)

    target = cv2.imread('target.png', 0)
    template = cv2.imread('template.png', 0)
    bounding_boxes = face_recognition(target, template, extract_hog)
    target_rgb = cv2.imread('target.png')
    visualize_face_detection(target_rgb, bounding_boxes, template.shape[0])
