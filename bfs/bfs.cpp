#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// #define VERBOSE
#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define NON_VERTEX -1

void vertex_set_clear(vertex_set* list) { list->count = 0; }

void vertex_set_init(vertex_set* list, int count) {
  list->max_vertices = count;
  list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set* frontier, vertex_set* new_frontier,
                   int* distances) {
  int distance = distances[frontier->vertices[0]];

#pragma omp parallel for schedule(static, 64)
  for (int i = 0; i < frontier->count; i++) {
    int node = frontier->vertices[i];

    int start_edge = g->outgoing_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->outgoing_starts[node + 1];
    int local_size = end_edge - start_edge;
    int local_set[local_size];
    int local_counter = 0;

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int outgoing = g->outgoing_edges[neighbor];
      if (distances[outgoing] == NOT_VISITED_MARKER &&
          __sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER,
                                       distance + 1)) {
        // we are responsible for this outgoing node.
        local_set[local_counter] = outgoing;
        ++local_counter;
      }
    }

    if (local_counter > 0) {
      int offset = __sync_fetch_and_add(&new_frontier->count, local_counter);
      memcpy(new_frontier->vertices + offset, &local_set,
             local_counter * sizeof(int));
    }
  }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("num_nodes=%d frontier=%-10d %.4f sec\n", graph->num_nodes,
           frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

void bottom_up_step(Graph graph, vertex_set* frontier, vertex_set* new_frontier,
                    int* distances) {
  static constexpr int kChunkSize = 256;
  int distance = distances[frontier->vertices[0]];
  int num_chunks = (graph->num_nodes + kChunkSize - 1) / kChunkSize;

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < num_chunks; ++i) {
    Vertex local_chunk[kChunkSize];
    int local_count = 0;

    for (int v = i * kChunkSize;
         v < std::min(graph->num_nodes, i * kChunkSize + kChunkSize); ++v) {
      if (distances[v] != NOT_VISITED_MARKER) continue;

      const Vertex* begin = incoming_begin(graph, v);
      int size = incoming_size(graph, v);
      if (size <= 0) continue;

      for (int off = 0; off < size; ++off) {
        Vertex p = begin[off];
        if (distances[p] == distance) {
          distances[v] = distance + 1;
          local_chunk[local_count++] = v;
          break;
        }
      }
    }

    int offset = __sync_fetch_and_add(&new_frontier->count, local_count);
    for (int i = 0; i < local_count; ++i) {
      new_frontier->vertices[offset + i] = local_chunk[i];
    }
  }
}

void bfs_bottom_up(Graph graph, solution* sol) {
  // CS149 students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for schedule(static, 64)
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    vertex_set_clear(new_frontier);
    bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("num_nodes=%d frontier=%-10d %.4f sec\n", graph->num_nodes,
           frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

uint64_t top_down_cost(Graph g, vertex_set* frontier) {
  uint64_t cost = 0;

#pragma omp parallel for reduction(+ : cost)
  for (int i = 0; i < frontier->count; ++i) {
    Vertex v = frontier->vertices[i];
    cost += outgoing_size(g, v);
  }

  return cost;
}

uint64_t bottom_up_cost(Graph g, int* distances) {
  uint64_t cost = 0;

#pragma omp parallel for reduction(+ : cost)
  for (int v = 0; v < g->num_nodes; ++v) {
    if (distances[v] != NOT_VISITED_MARKER) continue;
    cost += incoming_size(g, v);
  }

  return cost;
}

void bfs_hybrid(Graph graph, solution* sol) {
  // CS149 students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set* frontier = &list1;
  vertex_set* new_frontier = &list2;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for schedule(static, 64)
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  int num_remaining = graph->num_nodes - 1;

  while (frontier->count != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    vertex_set_clear(new_frontier);
    if (frontier->count >= num_remaining) {
      bottom_up_step(graph, frontier, new_frontier, sol->distances);
    } else {
      top_down_step(graph, frontier, new_frontier, sol->distances);
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("num_nodes=%d frontier=%-10d %.4f sec\n", graph->num_nodes,
           frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set* tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;

    num_remaining -= frontier->count;
  }
}
