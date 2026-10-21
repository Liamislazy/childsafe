// Optional real browser check. Synthetic state only; never requests a physical camera.
const {chromium}=require(process.env.CHILDSAFE_PLAYWRIGHT_MODULE || 'playwright');
const url=process.env.CHILDSAFE_TEST_URL || 'http://127.0.0.1:8767';
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(url);
    await page.waitForFunction(()=>info?.product_scope);
    const scope=await page.evaluate(()=>info.product_scope);
    if(scope.additional_detector_categories.length || !scope.pending_training_categories.includes('pen'))
      throw new Error('UI claimed a custom class without its trained weights');
    await page.evaluate(()=>{
      renderRuntime({level:0,green:false,objects:[],people:[{
        id:1,level:0,level_name:'normal',green:false,hands:[],mouth_visible:true,
        image_observation:'observable',validation_status:'unvalidated',object_scope:'limited',
        object_localized:true,object:'scissors',yolo_label:'scissors',mask_reliable:true,
        mask_temporally_stable:true,class:'other',status:'uncertain',hazard_status:'known_hazard',
        hazard_source:'detector',object_confidence:.9,d:.2,d_hand:.5,d_object:.2,d_source:'object',
        near:false,dwell:0,reasons:[]
      }],limitations:[],detector_ok:true,classifier_ok:true});
    });
    const state=await page.locator('#liveModels .level').evaluate(e=>({background:getComputedStyle(e).backgroundColor,text:e.textContent}));
    if(state.background!=='rgb(51, 65, 85)' || !state.text.includes('quan sát hạn chế'))
      throw new Error('Limited observation badge still looks green: '+JSON.stringify(state));
    if(!(await page.locator('#liveModels').textContent()).includes('detector độc lập'))
      throw new Error('Independent detector category source is missing');
    // A frame already presented during inference must be consumed without waiting
    // for a further video callback. No physical stream is used here.
    const immediate=await page.evaluate(async()=>{
      latestMediaFrame=15;lastMediaFrame=10;
      return await nextFrame({requestVideoFrameCallback(){throw new Error('extra camera wait')}});
    });
    if(!immediate || errors.length) throw new Error(JSON.stringify(errors));
    console.log(JSON.stringify({passed:true,physical_camera_opened:false,
      scope:'actual class inventory, limited badge, independent detector source, latest presented frame'}));
  } finally {await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
