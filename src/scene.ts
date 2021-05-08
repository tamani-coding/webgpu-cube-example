import { mat4, vec3 } from 'gl-matrix';
import { Cube } from './objects';

export const lightDataSize = 3 * 4; // 4x4 matrix size + vec3 size

export class Scene {

    public pointLightPosition = vec3.fromValues(0, 0, 0);

    private objects: Cube[] = [];

    public add (object: Cube) {
        this.objects.push(object);
    }

    public getObjects () : Cube[] {
        return this.objects;
    }

    public getPointLightPosition(): Float32Array {
        return this.pointLightPosition as Float32Array;
    }
}