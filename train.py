def train(epochs,lr,resume_epoch=0):

    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr,T_max=3700), warmup_steps=300, start_lr=0.00001, end_lr=lr)


    optim_segmentor = paddle.optimizer.Adam(learning_rate=scheduler, parameters=segmentor.parameters())
    optim_segmentor_feature = paddle.optimizer.Adam(learning_rate=scheduler, parameters=segmentor.parameters()[:96])

    
    for epoch in range(epochs):
        generator_src.eval()
        generator_tar.eval()
        segmentor.train()

        losses_ct_consistency=[]
        losses_segment=[]


        for j in range(0,50,1):
            images_source,labels_source=functions.pick_data(path='mmwhs/gt_train_{}/'.format(source_name),training=1,num_slices=16,num_foreground_slices=4)
            images_target,sda_images_target,_=functions.sda_pick_data(path='mmwhs/gt_train_{}/'.format(target_name),num_slices=16)



            #source generator
            source_fake=generator_src(images_source,mode='identity')

            #segmentation for MRI
            in_for_target=source_fake
            segment_fake=segmentor(in_for_target,mode='pure')

         
            loss_segment_fake1=dice_loss(y_pred=nn.functional.softmax(paddle.transpose(segment_fake,perm=[0,2,3,1]),axis=-1),y_true=nn.functional.one_hot(labels_source,num_classes=5))
            loss_segment_fake2=nn.functional.cross_entropy(input=paddle.transpose(segment_fake,perm=[0,2,3,1]),label=labels_source)
            loss_segment=(loss_segment_fake1+loss_segment_fake2)/2
            losses_segment.append(loss_segment.item())
          
            optim_segmentor.clear_grad()
            loss_segment.backward()
            optim_segmentor.step()

            #segmentation for CT
            segment_cons0=segmentor(images_target,mode='pure')
            segment_cons1=segmentor(sda_images_target,mode='pure')

            loss_ct_consistency=nn.functional.mse_loss(input=segment_cons0,label=segment_cons1)
            losses_ct_consistency.append(loss_ct_consistency.item())
            
            optim_segmentor_feature.clear_grad()
            loss_ct_consistency.backward()
            optim_segmentor_feature.step()     
         

            scheduler.step()
        '''if(epoch%10==0 and epoch!=0):

            paddle.save(segmentor.state_dict(),'model/segmentor{}'.format(epoch+resume_epoch))
            paddle.save(discriminator_feature.state_dict(),'model/discriminator_feature{}'.format(epoch+resume_epoch))
            [paddle.save(mlp_list_seg[i].state_dict(),'model/mlp_seg{}_{}'.format(i,epoch+resume_epoch)) for i in range(5)]'''

        
        print('epoch:{} consistency:{} segment:{}'.format(
            epoch+resume_epoch,
            np.mean(losses_ct_consistency),
            np.mean(losses_segment)
            ))
        file1=open(log_file,mode='a')
        file1.write('epoch:{} consistency:{} segment:{}\n'.format(
            epoch+resume_epoch,
            np.mean(losses_ct_consistency),
            np.mean(losses_segment)
            ))
        file1.close()



generator_src=newnet.Cyclegan(num_channels=num_channels_gen,in_channels=1)
generator_tar=newnet.Cyclegan(num_channels=num_channels_gen,in_channels=1)

segmentor=unet.UNet()
source_domain_name='ct'
if(source_domain_name=='ct'):
    generator_src.set_state_dict(paddle.load('generator_ct_to_mr'))
    generator_tar.set_state_dict(paddle.load('generator_mr_to_ct'))
    source_name='ct'
    target_name='mr'
    log_file='log_ct_to_mr.txt'
elif(source_domain_name=='mr'):
    generator_src.set_state_dict(paddle.load('generator_mr_to_ct'))
    generator_tar.set_state_dict(paddle.load('generator_ct_to_mr'))
    source_name='mr'
    target_name='ct'
    log_file='log_mr_to_ct.txt'
train(80,lr=0.0004,resume_epoch=0)
paddle.save(segmentor.state_dict(),'segmentor_{}_final'.format(source_domain_name))
