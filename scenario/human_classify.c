/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov3_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#include "mpi_sys.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define HAND_FRM_WIDTH     640
#define HAND_FRM_HEIGHT    384
#define DETECT_OBJ_MAX     32
#define RET_NUM_MAX        4
#define DRAW_RETC_THICK    2    // Draw the width of the line
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32
#define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT       224
#define MODEL_FILE_GESTURE    "/userdata/models/hand_classify/hand_gesture.wk" // darknet framework wk model

static int biggestBoxIndex;
static IVE_IMAGE_S img;
static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
static RectBox boxs[DETECT_OBJ_MAX] = {0};
static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;
int uartFd1 = 0;



HI_S32 Yolo3HandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;
    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    HumanDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load human detect claasify model success\n");

    uartFd1 = UartOpenInit();
    if (uartFd1 < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}



HI_S32 Yolo3HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    HumanDetectExit(); // Uninitialize the hand detection model
    SAMPLE_PRT("Unload human detect claasify model success\n");

    return 0;
}

/* Get the maximum hand */
static HI_S32 GetBiggestHandIndex(RectBox boxs[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
    HI_S32 biggestBoxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
        HI_S32 boxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = boxs[biggestBoxIndex].xmax - boxs[biggestBoxIndex].xmin + 1;
        biggestBoxHeight = boxs[biggestBoxIndex].ymax - boxs[biggestBoxIndex].ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}

/* hand gesture recognition info */
static void HandDetectFlag(const RecogNumInfo resBuf)
{
    HI_CHAR *gestureName = NULL;
    SAMPLE_PRT("human gesture success\n");
}

void ComposeSendData(int xmin, int ymin, int xmax, int ymax, char* data, int* size){
   sprintf(data, "%d %d %d %d\0", ymin, xmin, ymax, xmax);
   *size = 0;
   for(int i=0;i<21;i++){
       (*size)++;
       if(data[i] == '\0'){
           break;
       }
   }
}

int first_run = 1;
HI_S32 Yolo3HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int num = 0;

    //0528_修改帧格式
    if(first_run == 1){
        first_run = 0;
    }
    else{
        HI_MPI_SYS_MmzFree(img.au64PhyAddr[0], (void*)img.au64VirAddr[0]);
    }
    ret = FrmToRgbImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "human detect for YUV Frm to RgbImg FAIL, ret=%#x\n", ret);

    ret = ImgRgbToBgr(&img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "human detect for RgbImg to Bgr FAIL, ret=%#x\n", ret);

    //ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    //SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "human detect for YUV Frm to Img FAIL, ret=%#x\n", ret);
    
    /*
    //0525debug：img的来源是srcFrm，srcData的来源是img
    if(&img==NULL){
        printf("*************img=NULL***********\r\n");
    } else {
        printf("*************img!=NULL***********\r\n");
    }*/

    objNum = HumanDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        SAMPLE_PRT("yolo3_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;
    }
    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    SAMPLE_PRT("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    // When an object is detected, a rectangle is drawn in the DSTFRM
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        //printf("*****begin draw 1*****\n");
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_GREEN, DRAW_RETC_THICK); // Target human objnum is equal to 1
        //printf("*****end draw 1*****\n");
        
        char data[20] = {};
        int size = 0;
        RectBox *biggestbox = &boxs[biggestBoxIndex];
        //UartSendRead(uartFd1,biggestbox->xmin, biggestbox->ymin, biggestbox->xmax, biggestbox->ymax);
        //映射关系：top<-ymin, left<-xmin, bottom<-ymax, right<-xmax
        ComposeSendData(biggestbox->xmin, biggestbox->ymin, biggestbox->xmax, biggestbox->ymax, data, &size);
        printf("send detect status:%s\n", data);
        UartSend(uartFd1,data,size);
        IveImgDestroy(&imgIn);
    }

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

