import { Scene } from './scene';
import { Camera } from './camera';
import { lightDataSize } from './scene';

export var device: GPUDevice;
export var cameraUniformBuffer: GPUBuffer;
export var lightDataBuffer: GPUBuffer;

export class WebGpuRenderer {

    readonly swapChainFormat = 'bgra8unorm';
    private initSuccess: boolean = false;
    private renderPassDescriptor: GPURenderPassDescriptor;

    private context: GPUCanvasContext;
    private presentationFormat: GPUTextureFormat;
    private presentationSize: number[];

    private matrixSize = 4 * 16; // 4x4 matrix

    constructor() { }

    public async init(canvas: HTMLCanvasElement): Promise<boolean> {
        if (!canvas) {
            console.log('missing canvas!')
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter();
        device = await adapter.requestDevice();

        if (!device) {
            console.log('found no gpu device!')
            return false;
        }

        this.context = canvas.getContext('webgpu');

        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.presentationSize = [
            canvas.clientWidth * devicePixelRatio,
            canvas.clientHeight  * devicePixelRatio,
        ];

        this.context.configure({
            device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        this.renderPassDescriptor = {
            colorAttachments: [
                {
                    // attachment is acquired and set in render loop.
                    view: undefined,
                    loadOp: 'clear',
                    clearValue: { r: 0.25, g: 0.25, b: 0.25, a: 1.0 }, 
                    storeOp: 'store',
                } as GPURenderPassColorAttachment,
            ],
            depthStencilAttachment: {
                view: this.depthTextureView(canvas),

                depthLoadOp: 'clear',
                depthClearValue: 1.0,
                depthStoreOp: 'store',
                stencilLoadOp: 'clear',
                stencilClearValue: 0,
                stencilStoreOp: 'store',
            } as GPURenderPassDepthStencilAttachment,
        };

        cameraUniformBuffer = device.createBuffer({
            size: this.matrixSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        lightDataBuffer = device.createBuffer({
            size: lightDataSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        return this.initSuccess = true;
    }

    public update(canvas: HTMLCanvasElement) {
        if (!this.initSuccess) {
            return;
        }

        this.updateRenderPassDescriptor(canvas);
    }

    public frame(camera: Camera, scene: Scene) {
        if (!this.initSuccess) {
            return;
        }

        // CAMERA BUFFER
        const cameraViewProjectionMatrix = camera.getCameraViewProjMatrix() as Float32Array;
        device.queue.writeBuffer(
            cameraUniformBuffer,
            0,
            cameraViewProjectionMatrix.buffer,
            cameraViewProjectionMatrix.byteOffset,
            cameraViewProjectionMatrix.byteLength
        );

        // LIGHT BUFFER
        const lightPosition = scene.getPointLightPosition();
        device.queue.writeBuffer(
            lightDataBuffer,
          0,
          lightPosition.buffer,
          lightPosition.byteOffset,
          lightPosition.byteLength
        );

        (this.renderPassDescriptor.colorAttachments as [GPURenderPassColorAttachment])[0].view = this.context
            .getCurrentTexture()
            .createView();

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

        for (let object of scene.getObjects()) {
            object.draw(passEncoder, device)
        }

        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    private depthTextureView(canvas: HTMLCanvasElement) {
        return device.createTexture({
            size: [
                canvas.clientWidth, //* devicePixelRatio,
                canvas.clientHeight, //  * devicePixelRatio,
            ],
            format: 'depth24plus-stencil8',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        }).createView();
    }

    private updateRenderPassDescriptor(canvas: HTMLCanvasElement) {
        (this.renderPassDescriptor.depthStencilAttachment as GPURenderPassDepthStencilAttachment).view = this.depthTextureView(canvas);
    }
}