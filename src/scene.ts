import { Cube } from './objects';

export class Scene {

    private objects: Cube[] = [];

    public add (object: Cube) {
        this.objects.push(object);
    }

    public getObjects () : Cube[] {
        return this.objects;
    }
}