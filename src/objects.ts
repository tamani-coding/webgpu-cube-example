import { device, cameraUniformBuffer } from './renderer';
import { Camera } from './camera';
import { mat4, vec3 } from 'gl-matrix';


const vertices = [
    // front
    { pos: [-1, -1,  1], norm: [ 0,  0,  1], uv: [0, 0], },
    { pos: [ 1, -1,  1], norm: [ 0,  0,  1], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [ 0,  0,  1], uv: [0, 1], },
   
    { pos: [-1,  1,  1], norm: [ 0,  0,  1], uv: [0, 1], },
    { pos: [ 1, -1,  1], norm: [ 0,  0,  1], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 0,  0,  1], uv: [1, 1], },
    // right
    { pos: [ 1, -1,  1], norm: [ 1,  0,  0], uv: [0, 0], },
    { pos: [ 1, -1, -1], norm: [ 1,  0,  0], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 1,  0,  0], uv: [0, 1], },
   
    { pos: [ 1,  1,  1], norm: [ 1,  0,  0], uv: [0, 1], },
    { pos: [ 1, -1, -1], norm: [ 1,  0,  0], uv: [1, 0], },
    { pos: [ 1,  1, -1], norm: [ 1,  0,  0], uv: [1, 1], },
    // back
    { pos: [ 1, -1, -1], norm: [ 0,  0, -1], uv: [0, 0], },
    { pos: [-1, -1, -1], norm: [ 0,  0, -1], uv: [1, 0], },
    { pos: [ 1,  1, -1], norm: [ 0,  0, -1], uv: [0, 1], },
   
    { pos: [ 1,  1, -1], norm: [ 0,  0, -1], uv: [0, 1], },
    { pos: [-1, -1, -1], norm: [ 0,  0, -1], uv: [1, 0], },
    { pos: [-1,  1, -1], norm: [ 0,  0, -1], uv: [1, 1], },
    // left
    { pos: [-1, -1, -1], norm: [-1,  0,  0], uv: [0, 0], },
    { pos: [-1, -1,  1], norm: [-1,  0,  0], uv: [1, 0], },
    { pos: [-1,  1, -1], norm: [-1,  0,  0], uv: [0, 1], },
   
    { pos: [-1,  1, -1], norm: [-1,  0,  0], uv: [0, 1], },
    { pos: [-1, -1,  1], norm: [-1,  0,  0], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [-1,  0,  0], uv: [1, 1], },
    // top
    { pos: [ 1,  1, -1], norm: [ 0,  1,  0], uv: [0, 0], },
    { pos: [-1,  1, -1], norm: [ 0,  1,  0], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 0,  1,  0], uv: [0, 1], },
   
    { pos: [ 1,  1,  1], norm: [ 0,  1,  0], uv: [0, 1], },
    { pos: [-1,  1, -1], norm: [ 0,  1,  0], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [ 0,  1,  0], uv: [1, 1], },
    // bottom
    { pos: [ 1, -1,  1], norm: [ 0, -1,  0], uv: [0, 0], },
    { pos: [-1, -1,  1], norm: [ 0, -1,  0], uv: [1, 0], },
    { pos: [ 1, -1, -1], norm: [ 0, -1,  0], uv: [0, 1], },
   
    { pos: [ 1, -1, -1], norm: [ 0, -1,  0], uv: [0, 1], },
    { pos: [-1, -1,  1], norm: [ 0, -1,  0], uv: [1, 0], },
    { pos: [-1, -1, -1], norm: [ 0, -1,  0], uv: [1, 1], },
  ];

  const wgslShaders = {
    vertex: `
        [[block]] struct Uniforms {
            matrix : mat4x4<f32>;
        };
        
        [[binding(0), group(0)]] var<uniform> modelTransform : Uniforms;
        [[binding(1), group(0)]] var<uniform> cameraTransform : Uniforms;
        
        struct VertexOutput {
            [[builtin(position)]] Position : vec4<f32>;

            [[location(0)]] fragColor : vec4<f32>;
            [[location(1)]] norm : vec4<f32>;
            [[location(2)]] uv : vec2<f32>;
        };
        
        [[stage(vertex)]]
        fn main([[location(0)]] position : vec3<f32>,
                [[location(1)]] norm : vec3<f32>,
                [[location(2)]] uv : vec2<f32>) -> VertexOutput {
            return VertexOutput(
                    cameraTransform.matrix * modelTransform.matrix * vec4<f32>(position, 1.0),  // vertex position
                    vec4<f32>(0.8, 0.8, 0.0, 1.0),                                              // color
                    modelTransform.matrix * vec4<f32>(norm, 1.0),                               // norm vector
                    uv                                                                          // uv
            );
        }
  `,
    fragment: `
        [[stage(fragment)]]
        fn main([[location(0)]] fragColor : vec4<f32>,
                [[location(1)]] norm : vec4<f32>,
                [[location(2)]] uv : vec2<f32>) -> [[location(0)]] vec4<f32> {
            return fragColor;
        }
  `,
};

export interface CubeParameter {

    x?: number;
    y?: number;
    z?: number;

    rotX?: number;
    rotY?: number;
    rotZ?: number;

}

export class Cube {

    public x: number = 0;
    public y: number = 0;
    public z: number = 0;

    public rotX: number = 0;
    public rotY: number = 0;
    public rotZ: number = 0;

    private matrixSize = 4 * 16; // 4x4 matrix
    private offset = 256; // uniformBindGroup offset must be 256-byte aligned
    private uniformBufferSize = this.offset + this.matrixSize;

    private modelViewProjectionMatrix = mat4.create() as Float32Array;

    private renderPipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private verticesBuffer: GPUBuffer;

    private perVertex = ( 3 + 3 + 2 );      // 3 for position, 3 for normal, 2 for uv
    private stride = this.perVertex * 4;    // stride = byte length of vertex data array 

    constructor(parameter?: CubeParameter) {
        this.renderPipeline = device.createRenderPipeline({
            vertex: {
                module: device.createShaderModule({
                    code: wgslShaders.vertex,
                }),
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: this.stride,
                        attributes: [
                            {
                                // position
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x3',
                            },
                            {
                                // norm
                                shaderLocation: 1,
                                offset: 3 * 4,
                                format: 'float32x3',
                            },
                            {
                                // uv
                                shaderLocation: 2,
                                offset: (3 + 3) * 4,
                                format: 'float32x2',
                            },
                        ],
                    } as GPUVertexBufferLayout,
                ],
            },
            fragment: {
                module: device.createShaderModule({
                    code: wgslShaders.fragment,
                }),
                entryPoint: 'main',
                targets: [
                    {
                        format: 'bgra8unorm' as GPUTextureFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus-stencil8',
            },
        });

        this.uniformBuffer = device.createBuffer({
            size: this.uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.uniformBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer,
                        offset: 0,
                        size: this.matrixSize,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: cameraUniformBuffer,
                        offset: 0,
                        size: this.matrixSize,
                    },
                },
            ],
        });

        this.verticesBuffer = device.createBuffer({
            size: vertices.length * this.stride,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        const mapping = new Float32Array(this.verticesBuffer.getMappedRange());
        for (let i = 0; i < vertices.length; i++) {
            // (3 * 4) + (3 * 4) + (2 * 4)
            mapping.set(vertices[i].pos, this.perVertex * i + 0);
            mapping.set(vertices[i].norm, this.perVertex * i + 3);
            mapping.set(vertices[i].uv, this.perVertex * i + 6);
        }
        this.verticesBuffer.unmap();

        this.setTransformation(parameter);
    }

    public draw(passEncoder: GPURenderPassEncoder, device: GPUDevice) {
        this.updateTransformationMatrix()

        passEncoder.setPipeline(this.renderPipeline);
        device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            this.modelViewProjectionMatrix.buffer,
            this.modelViewProjectionMatrix.byteOffset,
            this.modelViewProjectionMatrix.byteLength
        );
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.draw(vertices.length, 1, 0, 0);
    }

    private updateTransformationMatrix() {
        // MOVE / TRANSLATE OBJECT
        const modelMatrix = mat4.create();
        mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(this.x, this.y, this.z))
        mat4.rotateX(modelMatrix, modelMatrix, this.rotX);
        mat4.rotateY(modelMatrix, modelMatrix, this.rotY);
        mat4.rotateZ(modelMatrix, modelMatrix, this.rotZ);

        // PROJECT ON CAMERA
        mat4.copy(this.modelViewProjectionMatrix, modelMatrix)
    }

    private setTransformation(parameter?: CubeParameter) {
        if (parameter == null) {
            return;
        }

        this.x = parameter.x ? parameter.x : 0;
        this.y = parameter.y ? parameter.y : 0;
        this.z = parameter.z ? parameter.z : 0;

        this.rotX = parameter.rotX ? parameter.rotX : 0;
        this.rotY = parameter.rotY ? parameter.rotY : 0;
        this.rotZ = parameter.rotZ ? parameter.rotZ : 0;
    }
}
