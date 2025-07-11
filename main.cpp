//
//  main.cpp
//  Zappy3DModel
//
//  Created by Mark Bonasoro on 2024-07-01.
//

#define GL_SILENCE_DEPRECATION

#include <GLUT/glut.h>
#include <iostream>
#include <vector>

#include "ufbx.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_WEIGHTS 4
#define TIMER_MS 16  // About 60 FPS

GLuint tex = -1;
typedef struct Vertex {
    ufbx_vec3 position;
    ufbx_vec3 normal;
    ufbx_vec2 texCoord;

    float weights[MAX_WEIGHTS];
    uint32_t bones[MAX_WEIGHTS];
} Vertex;

typedef struct Bone {
    uint32_t node_index;
    ufbx_matrix geometry_to_bone;
    ufbx_matrix rest_pose;
} Bone;

typedef struct Mesh {
    Bone* bones;
    size_t num_bones;
    Vertex* vertices;
    size_t num_vertices;
} Mesh;

Mesh mesh;
ufbx_scene* scene;
double anim_time = 0.0;

GLuint loadTexture() {
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load("Map__1.png", &width, &height, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    return texture;
}

Vertex get_skinned_vertex(ufbx_mesh* mesh, ufbx_skin_deformer* skin, size_t index) {
    Vertex v = { 0 };
    v.position = ufbx_get_vertex_vec3(&mesh->vertex_position, index);
    v.normal = ufbx_get_vertex_vec3(&mesh->vertex_normal, index);
    v.texCoord = ufbx_get_vertex_vec2(&mesh->vertex_uv, index);

    uint32_t vertex = mesh->vertex_indices.data[index];
    ufbx_skin_vertex skin_vertex = skin->vertices.data[vertex];
    size_t num_weights = skin_vertex.num_weights;
    if (num_weights > MAX_WEIGHTS) {
        num_weights = MAX_WEIGHTS;
    }

    float total_weight = 0.0f;
    for (size_t i = 0; i < num_weights; i++) {
        ufbx_skin_weight skin_weight = skin->weights.data[skin_vertex.weight_begin + i];
        v.bones[i] = skin_weight.cluster_index;
        v.weights[i] = (float)skin_weight.weight;
        total_weight += (float)skin_weight.weight;
    }

    for (size_t i = 0; i < num_weights; i++) {
        v.weights[i] /= total_weight;
    }

    return v;
}

Mesh process_skinned_mesh(ufbx_mesh* mesh, ufbx_skin_deformer* skin) {
    size_t num_triangles = mesh->num_triangles;
    Vertex* vertices = (Vertex*)calloc(num_triangles * 3, sizeof(Vertex));
    size_t num_vertices = 0;

    size_t num_tri_indices = mesh->max_face_triangles * 3;
    uint32_t* tri_indices = (uint32_t*)calloc(num_tri_indices, sizeof(uint32_t));
    for (size_t face_ix = 0; face_ix < mesh->num_faces; face_ix++) {
        ufbx_face face = mesh->faces.data[face_ix];
        uint32_t num_tris = ufbx_triangulate_face(tri_indices, num_tri_indices, mesh, face);
        for (size_t i = 0; i < num_tris * 3; i++) {
            uint32_t index = tri_indices[i];
            vertices[num_vertices++] = get_skinned_vertex(mesh, skin, index);
        }
    }
    free(tri_indices);
    assert(num_vertices == num_triangles * 3);

    size_t num_bones = skin->clusters.count;
    Bone* bones = (Bone*)calloc(num_bones, sizeof(Bone));
    for (size_t i = 0; i < num_bones; i++) {
        ufbx_skin_cluster* cluster = skin->clusters.data[i];
        bones[i].node_index = cluster->bone_node->typed_id;
        bones[i].geometry_to_bone = cluster->geometry_to_bone;

    }


    Mesh result = {
        result.bones = bones,
        result.num_bones = num_bones,
        result.vertices = vertices,
        result.num_vertices = num_vertices,
    };
    return result;
}

void matrix_add(ufbx_matrix* dst, const ufbx_matrix* src, float weight) {
    for (size_t i = 0; i < 3 * 4; i++) {
        dst->v[i] += src->v[i] * weight;
    }
}

void draw_mesh(Mesh mesh, ufbx_scene* scene) {
    ufbx_matrix* geometry_to_world = (ufbx_matrix*)calloc(mesh.num_bones, sizeof(ufbx_matrix));

    for (size_t i = 0; i < mesh.num_bones; i++) {
        Bone bone = mesh.bones[i];
        ufbx_node* node = scene->nodes.data[bone.node_index];
//        if(i == 0)
//        std::cout<<"t: "<<anim_time<<" Bone (0) ("<< node->local_transform.translation.x <<","<< node->local_transform.translation.y <<"," << node->local_transform.translation.z <<")"<<std::endl;
        ufbx_matrix bone_to_world = node->node_to_world;
        geometry_to_world[i] = ufbx_matrix_mul(&bone_to_world, &bone.geometry_to_bone);
    }

    glBegin(GL_TRIANGLES);
    for (size_t vertex_ix = 0; vertex_ix < mesh.num_vertices; vertex_ix++) {
        Vertex vertex = mesh.vertices[vertex_ix];
        ufbx_matrix vertex_to_world = { 0 };
        for (size_t i = 0; i < MAX_WEIGHTS; i++) {
            uint32_t bone_ix = vertex.bones[i];
            float weight = vertex.weights[i];
            matrix_add(&vertex_to_world, &geometry_to_world[bone_ix], weight);
        }

        ufbx_vec3 p = ufbx_transform_position(&vertex_to_world, vertex.position);
        ufbx_vec3 n = ufbx_transform_direction(&vertex_to_world, vertex.normal);
        ufbx_vec2 uv = vertex.texCoord; // Retrieve texture coordinates

        glNormal3f(n.x, n.y, n.z);
        glTexCoord2f(uv.x, uv.y); // Set texture coordinates
        glVertex3f(p.x, p.y, p.z);
        
//        if(vertex_ix == 2182)
//        {
//            std::cout<<"t: "<<anim_time<<" Vertex "<< vertex_ix <<" ("<< p.x <<","<< p.y <<"," << p.z <<")"<<std::endl;
//        }
    }
    glEnd();

    free(geometry_to_world);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    if(tex == -1)
        tex = loadTexture();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glPushMatrix();
    //glScaled(0.05f, 0.05f, 0.05f);
    //glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
    glTranslatef(0, -15, 0);

    draw_mesh(mesh, scene);
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
}

void timer(int value) {
    anim_time += TIMER_MS / 1000.0; // Increment animation time
    double endTime = scene->anim->time_end;
    if (anim_time > endTime) {
        anim_time = 0.0; // Loop animation
    }

    // Apply the animation at the current time
    ufbx_scene* updScene = ufbx_evaluate_scene(scene, scene->anim,anim_time, NULL,NULL);
    scene = updScene;
    
    glutPostRedisplay(); // Request a redraw
    glutTimerFunc(TIMER_MS, timer, 0); // Restart the timer
}

void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void init_glut() {
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    GLfloat ambient_light[] = { 0.2, 0.2, 0.2, 1.0 };
    GLfloat diffuse_light[] = { 0.8, 0.8, 0.8, 1.0 };
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
}

int main(int argc, char** argv) {
    ufbx_load_opts opts = { 0 };
    ufbx_error error;
    
    if(argc != 2)
    {
        std::cout<<"Program requires a model to be loaded."<<std::endl;
        
        return -1;
    }
    
    //scene = ufbx_load_file("Saru_Idle.fbx", &opts, &error);
    //scene = ufbx_load_file("Saru_Idle2.fbx", &opts, &error);
    scene = ufbx_load_file(argv[0], &opts, &error);
    //scene = ufbx_load_file("test_bones_animation.fbx", &opts, &error);
    assert(scene);

    ufbx_node* node = ufbx_find_node(scene, scene->nodes.data[1]->name.data);
    assert(node && node->mesh);

    ufbx_mesh* ufbxMesh = node->mesh;
    assert(ufbxMesh->skin_deformers.count > 0);
    ufbx_skin_deformer* skin = ufbxMesh->skin_deformers.data[0];

    mesh = process_skinned_mesh(ufbxMesh, skin);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("UFBX model loading with animation");

//    GLenum err = glewInit();
//    if (err != GLEW_OK) {
//        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
//        return 1;
//    }

    init_glut();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(TIMER_MS, timer, 0);  // Start the timer
    glutMainLoop();

    free(mesh.vertices);
    free(mesh.bones);
    ufbx_free_scene(scene);

    return 0;
}
