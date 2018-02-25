<!-- Contributed by David Bau, in the public domain -->

<template>
<div class="vectorlist">
  <div v-for="(vector, index) in vectors" class="vector">
    <input v-model="vector.text">
    <button @click="selectVector(index)">&rarr;</button>
    <button @click="deleteVector(index)">x</button>
  </div>
  <div class="operation">
  <button @click="saveVector()">Save current sample</button>
  </div>
  <div class="operation">
  <!-- TODO: Change this button to do something interesting -->
  <button @click="applyVectorMath()">Apply vector math</button>
  </div>
  <!-- TODO: Add the KNN font ID button below -->
</div>
</template>

<script>
import {Array1D, ENV} from 'deeplearn';

const math = ENV.math;

//This json file includes all of the Font IDs in our database and their 40-dimensional logits vector.
var json = require('../embeddings.json');

export default {
  props: {
    selectedSample: { },
    model: { },
    vectors: { type: Array, default: () => [ { text: "0" } ] }
  },
  methods: {
    saveVector() {
      this.selectedSample.data().then(x =>
         this.vectors.push({ text: Array.prototype.slice.call(x).join(',') })
      );
    },
    deleteVector(index) {
      this.vectors.splice(index, 1);
    },
    selectVector(index) {
      this.$emit("select", { selectedSample: this.model.fixdim(
           Array1D.new(this.vectors[index].text.split(',').map(parseFloat)))});
    },
    // TODO: Add useful vector space operations here -->
    applyVectorMath() {
      this.$emit("select", { selectedSample:
           math.add(this.selectedSample, this.model.fixdim(
               Array1D.new([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))) } )
    },

    //TODO: Implement getKNN to output the font ID of the nearest neighbor
    getKNN() {

    }

  },
  watch: {
    model: function(val) {
      for (let i = 0; i < this.vectors.length; ++i) {
        let arr = this.vectors[i].text.split(',');
        if (arr.length > this.model.dimensions) {
            arr = arr.slice(0, this.model.dimensions);
        }
        while (arr.length < this.model.dimensions) {
            arr.push('0');
        }
        this.vectors[i].text = arr.join(',');
      }
    }
  },
}
</script>

<style scoped>
.vector, .operation {
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  white-space: nowrap;
}

</style>
